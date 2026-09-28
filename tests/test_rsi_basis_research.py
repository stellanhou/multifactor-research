"""Hand-calculated RSI and paired-account evidence, plus research handoff."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from crypto_quant.features.factor_expressions import compile_expression, evaluate_expression, wilder_rsi
from crypto_quant.features.factor_inputs import FactorInputPanel
from crypto_quant.research.factor_mining.contracts import digest
from crypto_quant.research.strategy_research import basis_strategy as basis
from crypto_quant.research.strategy_research.cli import ReplayModel
from crypto_quant.research.strategy_research.contracts import StrategyResearchContract
from crypto_quant.research.strategy_research.engine import execution_plan, load_segment, DataGap
from crypto_quant.research.strategy_research.workflow import StrategyResearch, validate_run
from crypto_quant.research.strategy_research.rule_strategy import generate_rule_targets
from crypto_quant.strategies.relative_strength import run_relative_strength_backtest
from crypto_quant.backtesting.config import BacktestConfig

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples/strategy_research'


def contract():
    raw = json.loads((EXAMPLES / 'contract.json').read_text())
    raw.update(development_start='2026-04-01T00:00:00Z', validation_start='2026-04-04T00:00:00Z',
               validation_end='2026-04-07T00:00:00Z', imputed_hours=[])
    raw['task']['strategy_family'] = basis.FAMILY
    raw['parameter_space'] = {'symbol': 'BTCUSDT', 'margin_guard_ratio': 0.1,
                             'entry_basis': [0.01, 0.02], 'exit_basis': [0.001], 'capital_fraction': [0.25]}
    raw['costs'].update(fee_bps=0, slippage_bps=0, min_trade_fraction=0)
    return StrategyResearchContract.from_dict(raw)


def bars(index, opens, closes):
    return pd.DataFrame({'open': opens, 'high': np.maximum(opens, closes)*1.01,
                         'low': np.minimum(opens, closes)*.99, 'close': closes, 'volume': 10000.,
                         'quote_volume': 1000000., 'close_time': index+pd.Timedelta(hours=1)-pd.Timedelta(milliseconds=1),
                         'synthetic': False, 'shortened': False}, index=index)


def replies_for(parameters, c):
    replies = json.loads((EXAMPLES / 'factor_rules/multi-factor.replay.json').read_text())
    plan = execution_plan(parameters, c)
    replies[0]['response'].update(parameters=parameters, calculation_meaning=plan)
    replies[1]['response']['plan_sha256'] = digest(plan)
    return replies


class RSITests(unittest.TestCase):
    def test_rsi_threshold_boundaries_and_next_hour_fills(self):
        # Isolate execution from indicator arithmetic (Wilder arithmetic is tested below).
        index = pd.date_range('2026-04-01', periods=24, freq='h', tz='UTC')
        frames = {s: bars(index, np.full(24, 100.), np.full(24, 100.))
                  for s in ('BTCUSDT', 'ETHUSDT')}
        rsi = pd.DataFrame(29., index=index, columns=['BTCUSDT', 'ETHUSDT'])
        rsi.iloc[18:] = np.array([30., 29., 31., 50., 51., 51.])[:, None]
        definition = json.loads((EXAMPLES / 'rsi_basis/rsi.replay.json').read_text())[0]['response']['parameters']
        with patch('crypto_quant.features.factor_expressions.wilder_rsi', return_value=rsi):
            targets, trace = generate_rule_targets(frames, definition, 168, index[18])
        self.assertEqual(trace['entry'].loc[index[18], 'BTCUSDT'], 0)  # 29 -> 30
        self.assertGreater(trace['entry'].loc[index[20], 'BTCUSDT'], 0)  # 29 -> 31
        self.assertEqual(trace['exit'].loc[index[21], 'BTCUSDT'], 0)  # exactly 50
        for weights in targets.values():
            self.assertEqual(weights.loc[index[18]], 0)
            self.assertEqual(weights.loc[index[20]], .4)
            self.assertEqual(weights.loc[index[21]], .4)
            self.assertEqual(weights.loc[index[22]], 0)
        market = {s: f.loc[index[17]:] for s, f in frames.items()}
        weights = {s: t.loc[index[17]:] for s, t in targets.items()}
        result = run_relative_strength_backtest(market, weights, config=BacktestConfig(
            initial_capital=10000, fee_bps=0, slippage_bps=0, min_trade_fraction=0))
        self.assertEqual(result.weights.loc[index[20], 'trade_notional'], 0)
        self.assertGreater(result.weights.loc[index[21], 'trade_notional'], 0)
        self.assertEqual(result.weights.loc[index[22], 'trade_notional'], 0)
        self.assertGreater(result.weights.loc[index[23], 'trade_notional'], 0)
        self.assertEqual(len(result.trades), 2)
        self.assertTrue((result.trades['exit_time'] == index[23]).all())

    def test_wilder_seed_recursion_and_degenerate_cases(self):
        x = pd.DataFrame({'mixed': [1., 2., 1., 3.], 'up': [1., 2., 3., 4.],
                          'down': [4., 3., 2., 1.], 'flat': [1., 1., 1., 1.]})
        r = wilder_rsi(x, 2)
        self.assertTrue(r.iloc[:2].isna().all().all())
        self.assertEqual(r.mixed.iloc[2], 50)
        self.assertAlmostEqual(r.mixed.iloc[3], 100*1.25/(1.25+.25))
        self.assertEqual(r.up.iloc[-1], 100)
        self.assertEqual(r.down.iloc[-1], 0)
        self.assertEqual(r.flat.iloc[-1], 50)

    def test_gaps_restart_seed_and_future_changes_are_causal(self):
        x = pd.DataFrame({'price': [1., 2., 1., np.nan, 10., 11., 12., 11.]})
        r = wilder_rsi(x, 2)
        self.assertTrue(r.iloc[3:6].isna().all().all())
        self.assertEqual(r.price.iloc[6], 100)
        changed = x.copy()
        changed.iloc[7] = 999
        pd.testing.assert_frame_equal(r.iloc[:7], wilder_rsi(changed, 2).iloc[:7])

    def test_expression_seed_history_and_multiple_assets(self):
        idx = pd.MultiIndex.from_product([pd.date_range('2026-01-01', periods=4, freq='h', tz='UTC'),
                                         ['BTCUSDT','ETHUSDT']], names=['timestamp','symbol'])
        panel = FactorInputPanel(pd.DataFrame({'spot_close':[1.,4.,2.,3.,1.,2.,3.,1.]}, index=idx),
                                 pd.Series(True, index=idx), {})
        expression = compile_expression('ts_delay(ts_rsi(spot_close,2),1)')
        self.assertEqual(expression.lookback_hours, 3)
        self.assertIn('recursive', next(step['history'] for step in expression.calculation_steps() if step.get('operator') == 'ts_rsi'))
        r = evaluate_expression('ts_rsi(spot_close,2)',panel).values.unstack('symbol')
        self.assertAlmostEqual(r.BTCUSDT.iloc[-1], 100*1.25/1.5)
        self.assertEqual(r.ETHUSDT.iloc[-1], 0)


class BasisTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.contract = contract()
        self.parameters = {'entry_basis': .01, 'exit_basis': .001, 'capital_fraction': .25}

    def small_frames(self):
        index = pd.date_range('2026-03-31T23:00Z', periods=4, freq='h')
        spot = bars(index, np.full(4,100.), np.full(4,100.))
        perp = bars(index, np.array([102.,102.,100.,100.]), np.array([102.,100.,100.,100.]))
        mark = bars(index, np.full(4,102.), np.full(4,102.))
        times = [index[1],index[1]+pd.Timedelta(milliseconds=1),index[2],index[2]+pd.Timedelta(milliseconds=1)]
        funding = pd.DataFrame({'funding_rate':[.01,.001,.001,.01], 'mark_price':[102.,102.,101.,101.],
                                'funding_interval_hours': [8]*4}, index=pd.DatetimeIndex(times))
        return {'spot':spot,'perp':perp,'mark':mark,'funding':funding}

    def run_small(self, frames, fees=0, slip=0):
        costs={k:v for k,v in self.contract.costs.items() if k!='stress_multiplier'}
        costs.update(fee_bps=fees,slippage_bps=slip)
        return basis.run_account(frames,self.parameters,self.contract,pd.Timestamp(self.contract.development_start),costs,self.root)

    def test_cash_collateral_equal_legs_and_exact_funding_event_order(self):
        result, ledger = self.run_small(self.small_frames())
        self.assertEqual(ledger.cash.iloc[0],4950)
        self.assertEqual(ledger.spot_units.iloc[0],25)
        self.assertEqual(ledger.perp_units.iloc[0],-25)
        self.assertAlmostEqual(ledger.collateral_cash.iloc[0],2552.55)
        self.assertAlmostEqual(ledger.funding_cashflow.sum(),5.075)
        self.assertAlmostEqual(result.equity.iloc[-1],10055.075)
        payments=pd.read_csv(self.root/'funding-payments.csv')
        np.testing.assert_allclose(payments.cashflow,[0,2.55,2.525,0])
        self.assertAlmostEqual(result.trades.net_pnl.iloc[0],55.075)

    def test_both_leg_fees_and_slippage_reduce_equity(self):
        result, ledger = self.run_small(self.small_frames(),fees=10)
        self.assertAlmostEqual(ledger.fee.sum(),10.05)
        self.assertAlmostEqual(result.equity.iloc[-1],10045.025)
        self.assertAlmostEqual(result.trades.fees.iloc[0],10.05)
        result2, ledger2 = self.run_small(self.small_frames(),fees=10,slip=5)
        self.assertLess(result2.equity.iloc[-1],result.equity.iloc[-1])
        self.assertGreater(ledger2.slippage_cost.sum(),0)

    def test_perp_mark_changes_unrealized_pnl_and_future_cannot_change_past(self):
        frames=self.small_frames()
        frames['mark'].iloc[1,frames['mark'].columns.get_loc('close')]=103.
        _,ledger=self.run_small(frames)
        self.assertAlmostEqual(ledger.perp_unrealized_pnl.iloc[0],-25)
        frames['perp'].iloc[-1,frames['perp'].columns.get_loc('close')]=1000.
        _,later=self.run_small(frames)
        pd.testing.assert_frame_equal(ledger.iloc[:2],later.iloc[:2])

    def test_margin_guard_rejects_instead_of_inventing_liquidation_fill(self):
        frames=self.small_frames()
        frames['mark'].iloc[1,frames['mark'].columns.get_loc('high')]=210
        with self.assertRaisesRegex(ValueError,'margin guard breached'):
            self.run_small(frames)
        self.assertTrue((self.root/'margin-breach.json').exists())
        self.assertTrue((self.root/'partial-funding.csv').exists())

    def data_fixture(self):
        index=pd.date_range('2026-03-31', '2026-04-07',freq='h',tz='UTC')
        prices=np.full(len(index),100.)
        basis_price=100+2*(np.arange(len(index))%24<12)
        self.market={'spot':bars(index,prices,prices),'perp':bars(index,basis_price,basis_price),
                     'mark':bars(index,basis_price,basis_price)}
        fidx=pd.date_range('2026-03-30','2026-04-07',freq='8h',tz='UTC')
        self.funding=pd.DataFrame({'funding_rate':.0001,'funding_interval_hours':8,'mark_price':101.,'source_path':'fixture'},index=fidx)

    def loader(self,market,symbol,interval,price_type,start,end):
        key='spot' if market=='spot' else ('mark' if price_type=='mark' else 'perp')
        return self.market[key].loc[start:end].copy()

    def funding_loader(self,symbol,start,end,include_previous):
        selected=self.funding.loc[start:end]
        previous=self.funding.loc[self.funding.index<start].tail(1)
        return pd.concat([previous,selected]).copy()

    def test_complete_basis_research_and_independent_validation(self):
        self.data_fixture()
        # Real funding exports mix whole-second and millisecond event stamps.
        self.funding.index = pd.DatetimeIndex([
            t + pd.Timedelta(milliseconds=1 if i % 2 else 0)
            for i, t in enumerate(self.funding.index)])
        idea=json.loads((EXAMPLES/'idea.json').read_text())
        runner=StrategyResearch(self.contract,idea,ReplayModel(replies_for(self.parameters,self.contract)),
                                self.root/'unused.sqlite',self.root/'run',model_mode='replay')
        with patch('crypto_quant.research.strategy_research.workflow.data_catalog', return_value={'sources': {}}), \
             patch.object(basis.MarketDataStore,'load_bars',side_effect=self.loader), \
             patch.object(basis.MarketDataStore,'load_funding',side_effect=self.funding_loader):
            self.assertEqual(runner.run()['status'],'retained_for_validation')
            self.assertFalse((runner.root/'validation-access.json').exists())
            self.assertEqual(validate_run(runner.root,self.root/'unused.sqlite')['status'],'engineering_complete')
        snapshot = pd.read_csv(runner.root/'development_data/funding.csv', index_col=0)
        snapshot.index = pd.to_datetime(snapshot.index, format='ISO8601', utc=True).as_unit('ns')
        expected = self.funding.loc[snapshot.index].copy()
        expected.index = expected.index.as_unit('ns')
        pd.testing.assert_frame_equal(snapshot, expected, check_names=False, check_freq=False)
        self.assertTrue((snapshot.index.microsecond != 0).any())
        self.assertTrue((runner.root/'v0001/strategy/ledger.csv').exists())
        self.assertTrue((runner.root/'development_data/funding.csv').exists())
        request=json.loads(json.loads((runner.root/'model_calls/call-0001.request.json').read_text())[1]['content'])
        self.assertIn('entry_basis',request['strategy_definition_format'])

    def test_missing_funding_or_mark_is_not_silently_substituted(self):
        self.data_fixture()
        with patch.object(basis.MarketDataStore,'load_bars',side_effect=self.loader), \
             patch.object(basis.MarketDataStore,'load_funding',side_effect=self.funding_loader):
            self.funding=self.funding.drop(self.funding.index[5])
            with self.assertRaisesRegex(DataGap,'missing funding'):
                load_segment(self.root/'unused.sqlite',self.contract,'development')
            self.data_fixture()
            self.funding.loc[pd.Timestamp('2026-04-01T08:00Z'),'mark_price']=np.nan
            with self.assertRaisesRegex(DataGap,'settlement mark prices'):
                load_segment(self.root/'unused.sqlite',self.contract,'development')
