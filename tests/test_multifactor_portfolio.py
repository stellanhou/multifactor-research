import numpy as np
import pandas as pd
import pytest

from crypto_quant.research.strategy_research.multifactor_account import run_perpetual_account
from crypto_quant.research.strategy_research.multifactor_portfolio import (
    generate_score_weighted_targets,
    generate_tapered_targets,
    smooth_target_weights,
)


def _frame(values, columns=None):
    return pd.DataFrame(values, columns=columns,
                        index=pd.date_range('2024-01-01T00:00:00Z', periods=len(values), freq='24h'))


def test_tapered_rank_crossing_trades_only_weight_difference():
    scores = _frame([list(range(10, 0, -1)), [10, 8, 9, 7, 6, 5, 4, 3, 2, 1]], list('ABCDEFGHIJ'))
    target = generate_tapered_targets(scores, side_count=5, gross_exposure=.8, max_asset_weight=.2)
    np.testing.assert_allclose(target.iloc[0], np.array([5,4,3,2,1,-1,-2,-3,-4,-5]) * .4 / 15)
    assert target.iloc[1].B > 0 and target.iloc[1].C > 0
    assert target.iloc[1].B - target.iloc[0].B == pytest.approx(-.4/15)
    assert target.diff().iloc[1].abs().sum() == pytest.approx(2*.4/15)
    np.testing.assert_allclose(target.sum(axis=1), 0, atol=1e-15)


def test_tapered_shortage_and_caps_keep_cash_and_disjoint_sides():
    scores = _frame([[3,2,1,np.nan],[np.nan]*4], list('ABCD'))
    target = generate_tapered_targets(scores, side_count=5, gross_exposure=.8, max_asset_weight=.1)
    np.testing.assert_allclose(target.iloc[0], [.1,0,-.1,0])
    assert target.iloc[1].eq(0).all()


def test_score_weights_center_redistribute_caps_and_stay_neutral():
    scores = _frame([[109,101,99,91]], list('ABCD'))
    target = generate_score_weighted_targets(scores, gross_exposure=.8, max_asset_weight=.2)
    np.testing.assert_allclose(target.iloc[0], [.2,.2,-.2,-.2])
    shifted = generate_score_weighted_targets(scores-100, gross_exposure=.8, max_asset_weight=.2)
    pd.testing.assert_frame_equal(target, shifted)
    uncapped = generate_score_weighted_targets(scores, gross_exposure=.8, max_asset_weight=.4)
    np.testing.assert_allclose(uncapped.iloc[0], [.36,.04,-.04,-.36])


def test_score_weights_flat_cash_and_limited_side_capacity():
    scores = _frame([[1,1,1,1],[9,0,0,0],[np.nan]*4], list('ABCD'))
    target = generate_score_weighted_targets(scores, gross_exposure=.8, max_asset_weight=.2)
    assert target.iloc[[0,2]].eq(0).all().all()
    np.testing.assert_allclose(target.iloc[1], [.2,-.2/3,-.2/3,-.2/3])


def test_score_weights_respond_gradually_away_from_flat_cross_sections():
    scores = _frame([[3,2,1,-1,-2,-3],[3,2.01,1,-1,-2,-3]], list('ABCDEF'))
    target = generate_score_weighted_targets(scores, gross_exposure=.8, max_asset_weight=.2)
    assert target.diff().iloc[1].abs().sum() < .003
    np.testing.assert_allclose(target.sum(axis=1),0,atol=1e-15)


def test_target_ema_hand_path_and_forced_exit_reset():
    target = _frame([[.2,-.2],[.2,-.2],[0,0],[.2,-.2],[0,0],[.2,-.2]], ['A','B'])
    eligible = target.notna(); eligible.iloc[4] = False
    result = smooth_target_weights(target, eligible, alpha=.5)
    np.testing.assert_allclose(result.A, [.1,.15,.075,.1375,0,.1])
    np.testing.assert_allclose(result.B, -result.A)
    assert result.abs().sum(axis=1).le(.4).all()
    pd.testing.assert_frame_equal(smooth_target_weights(target, eligible, alpha=1), target)


def test_target_ema_asset_specific_ineligibility_and_future_invariance():
    target = _frame([[.2,-.2],[0,-.2],[.2,0],[0,0]], ['A','B'])
    eligible = target.notna(); eligible.iloc[1,0] = False
    result = smooth_target_weights(target,eligible,alpha=.5)
    assert result.iloc[1].A == 0
    changed = target.copy(); changed.iloc[2:] = [-.2,.2]
    pd.testing.assert_frame_equal(result.iloc[:2],smooth_target_weights(changed,eligible,alpha=.5).iloc[:2])


@pytest.mark.parametrize('method', ['tapered','score','ema'])
def test_continuous_targets_execute_without_rank_projection(method):
    scores = _frame([list(range(10,0,-1))],list('ABCDEFGHIJ'))
    if method in {'tapered','ema'}:
        targets = generate_tapered_targets(scores,side_count=5,gross_exposure=.8,max_asset_weight=.2)
    else:
        targets = generate_score_weighted_targets(scores,gross_exposure=.8,max_asset_weight=.2)
    if method == 'ema':
        targets = smooth_target_weights(targets,scores.notna(),alpha=.5)
    grid = pd.date_range(scores.index[0],periods=3,freq='h')
    frames={s:pd.DataFrame({'open':100.,'close':100.,'mark_close':100.},index=grid) for s in scores.columns}
    funding=pd.DataFrame(columns=['timestamp','symbol','funding_rate','mark_price'])
    account=run_perpetual_account(frames,targets,funding,initial_capital=1000.,fee_bps=10.,slippage_bps=0.,
                                 start=grid[1],end=grid[-1]+pd.Timedelta(hours=1),margin_fraction=.1)
    assert account.fills.symbol.nunique()==10
    assert account.fills.notional.sum()==pytest.approx(400 if method=='ema' else 800)
    assert account.metrics['total_fees']==pytest.approx(.4 if method=='ema' else .8)
    np.testing.assert_allclose(account.orders.set_index('symbol').target_weight.loc[scores.columns],targets.iloc[0])


@pytest.mark.parametrize('alpha',[0,1.1,np.nan])
def test_invalid_ema_alpha_rejected(alpha):
    target=_frame([[.2]],['A'])
    with pytest.raises(ValueError,match='alpha'):
        smooth_target_weights(target,target.notna(),alpha=alpha)


def test_invalid_eligibility_and_nonfinite_targets_rejected():
    target=_frame([[.2]],['A'])
    with pytest.raises(ValueError,match='ineligible'):
        smooth_target_weights(target,pd.DataFrame(False,index=target.index,columns=target.columns),alpha=.5)
    with pytest.raises(ValueError,match='exactly'):
        smooth_target_weights(target,target.notna().rename(columns={'A':'B'}),alpha=.5)
    with pytest.raises(ValueError,match='finite'):
        smooth_target_weights(target*np.nan,target.notna(),alpha=.5)
