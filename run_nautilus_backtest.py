#!/usr/bin/env python3
"""
Storm Tang V2 - Nautilus 回测入口 (GitHub Actions 版)
用法: python run_nautilus_backtest.py --start 2024-01-01 --end 2026-06-30
"""
import argparse
import sys
import os
from datetime import date, timedelta

# 添加工作目录
sys.path.insert(0, "/work")

import pandas as pd
import numpy as np
import time

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.model.enums import AccountType, OmsType, BarAggregation, PriceType
from nautilus_trader.model.identifiers import Venue, InstrumentId, Symbol
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.currencies import USDT, BTC
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.instruments import CryptoPerpetual
from decimal import Decimal

from strategy_v2_nautilus import StormV2Pure, StormV2Config

DATA_ROOT = "/data/crypto_data/ticks/BTCUSDT"

def load_bars(tf_seconds, start_d, end_d):
    """从 parquet 加载并聚合为 Bar"""
    bars_list = []
    d = start_d
    while d <= end_d:
        path = f"{DATA_ROOT}/BTCUSDT-aggTrades-{d.isoformat()}.parquet"
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            rule = f'{tf_seconds//60}min' if tf_seconds < 3600 else f'{tf_seconds//3600}h'
            ohlcv = df['price'].resample(rule).ohlc()
            vol = df['quantity'].resample(rule).sum().rename('volume')
            bars = pd.concat([ohlcv, vol], axis=1).dropna()
            bars.columns = ['open', 'high', 'low', 'close', 'volume']
            bars_list.append(bars.reset_index(drop=True))
        d += timedelta(days=1)
    return pd.concat(bars_list, ignore_index=True) if bars_list else pd.DataFrame()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD")
    args = parser.parse_args()

    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(args.end)

    print(f"=== Storm Tang V2 回测 ===")
    print(f"区间: {start_d} 到 {end_d}")
    print(f"数据源: {DATA_ROOT}")

    # 1. 加载 15M + 1H Bar 数据
    print("加载数据...")
    t0 = time.time()
    bars_15m = load_bars(900, start_d, end_d)
    bars_1h = load_bars(3600, start_d, end_d)
    print(f"15M: {len(bars_15m)}, 1H: {len(bars_1h)}, 耗时: {time.time()-t0:.1f}s")

    if len(bars_15m) == 0:
        print("ERROR: 无数据，退出")
        return

    # 2. 创建回测引擎
    engine = BacktestEngine(config=BacktestEngineConfig())

    # 3. 添加交易所
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(5000.0, USDT)],
    )

    # 4. 创建合约
    instrument = CryptoPerpetual(
        instrument_id=InstrumentId(Symbol("BTCUSDT-PERP"), Venue("BINANCE")),
        raw_symbol=Symbol("BTCUSDT"),
        base_currency=BTC,
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=2,
        size_precision=3,
        price_increment=Price.from_str("0.10"),
        size_increment=Quantity.from_str("0.001"),
        max_quantity=Quantity.from_str("1000"),
        min_quantity=Quantity.from_str("0.001"),
        max_notional=None,
        min_notional=Money(5, USDT),
        max_price=Price.from_str("1000000"),
        min_price=Price.from_str("0.10"),
        margin_init=Decimal("0.05"),
        margin_maint=Decimal("0.025"),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0004"),
        ts_event=0,
        ts_init=0,
        multiplier=Quantity.from_str("1"),
    )
    engine.add_instrument(instrument)

    # 5. 加载 15M Bar 数据
    print("转换 Bar 数据...")
    bar_type = BarType(
        instrument_id=instrument.id,
        spec=BarSpecification(15, BarAggregation.MINUTE, PriceType.LAST),
    )
    bars_list = []
    for _, row in bars_15m.iterrows():
        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(str(row['open'])),
            high=Price.from_str(str(row['high'])),
            low=Price.from_str(str(row['low'])),
            close=Price.from_str(str(row['close'])),
            volume=Quantity.from_str(str(row['volume'])),
            ts_event=int(row['timestamp'].timestamp() * 1_000_000),
            ts_init=int(row['timestamp'].timestamp() * 1_000_000),
        )
        bars_list.append(bar)

    print(f"加载 {len(bars_list)} 根 Bar...")
    engine.add_data(bars_list)

    # 6. 创建策略
    strategy = StormV2Pure(
        config=StormV2Config(
            instrument_id=str(instrument.id),
            bar_type=bar_type,
        )
    )
    engine.add_strategy(strategy)

    # 7. 运行回测
    print("开始回测...")
    t0 = time.time()
    engine.run()
    print(f"回测耗时: {time.time()-t0:.1f}s")

    # 8. 输出结果
    account = engine.cache.account_for_venue(Venue("BINANCE"))
    if account:
        for currency, balance in account.balances().items():
            final = float(balance.total)
            pnl = final - 5000.0
            print(f"\n=== 回测结果 ===")
            print(f"最终权益: {final:.2f} USDT")
            print(f"盈亏: {pnl:+.2f} USDT ({pnl/50:.1f}%)")

    fills = engine.cache.fills()
    orders = engine.cache.orders()
    print(f"成交笔数: {len(fills)}")
    print(f"订单数: {len(orders)}")

    # 保存结果
    os.makedirs("/work/backtest_results", exist_ok=True)
    import json
    result = {
        "start": args.start,
        "end": args.end,
        "final_equity": final if 'final' in locals() else 0,
        "pnl": pnl if 'pnl' in locals() else 0,
        "total_fills": len(fills),
        "total_orders": len(orders),
    }
    with open("/work/backtest_results/result.json", "w") as f:
        json.dump(result, f, indent=2)

    engine.dispose()
    print("回测完成")

if __name__ == "__main__":
    main()