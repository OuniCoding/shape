from pymodbus.server import StartTcpServer
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusDeviceContext,
    ModbusServerContext
)
import threading
import time

# =========================
# DataBlock
# =========================
store = ModbusDeviceContext(
    di=ModbusSequentialDataBlock(0, [0]*100),
    co=ModbusSequentialDataBlock(0, [0]*100),
    hr=ModbusSequentialDataBlock(0, [0]*100),
    ir=ModbusSequentialDataBlock(0, [0]*100),
)

context = ModbusServerContext(devices=store, single=True)

# =========================
# PLC 行為（不做多餘控制）
# =========================
def plc_logic():
    print("PLC 模擬啟動（獨立控制版）")

    last_m200 = 0
    last_m201 = 0
    last_m203 = 0

    while True:
        try:
            coils = context[0].getValues(1, 0, count=5)

            M200 = coils[0]
            M201 = coils[1]
            M202 = coils[2]
            M203 = coils[3]
            M204 = coils[4]

            # ====== Start ======
            if M200 == 1 and last_m200 == 0:
                print("PLC: Start Trigger")
                context[0].setValues(1, 2, [1])  # M202 ON（剔除完成）

            # ====== Stop ======
            if M201 == 1 and last_m201 == 0:
                print("PLC: Stop Trigger")

            # ====== Reset ======
            if M203 == 1 and last_m203 == 0:
                print("PLC: Reset Trigger")
                context[0].setValues(1, 4, [1])  # M204 ON（歸零完成）

            # ====== 更新狀態（做邊緣觸發用）
            last_m200 = M200
            last_m201 = M201
            last_m203 = M203

        except Exception as e:
            print("PLC Error:", e)

        time.sleep(0.1)

# =========================
# 手動輸入 D202/D203
# =========================
def manual_input():
    while True:
        try:
            val = input("請輸入 D202 (32bit 整數): ")

            val = int(val)

            low = val & 0xFFFF
            high = (val >> 16) & 0xFFFF

            context[0].setValues(3, 2, [low, high])

            print(f"已寫入 D202/D203 = {val}")

        except:
            print("輸入錯誤")

# =========================
# 啟動
# =========================
if __name__ == "__main__":
    threading.Thread(target=plc_logic, daemon=True).start()
    threading.Thread(target=manual_input, daemon=True).start()

    print("Modbus TCP Slave 啟動 (Port 5020)")
    StartTcpServer(context=context, address=("0.0.0.0", 5020))
'''
pymodbus 2.x
#----
from pymodbus.server import StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
import threading
import time

# =========================
# 建立資料區
# =========================
store = ModbusSlaveContext(
    di=None,
    co=ModbusSequentialDataBlock(0, [0]*100),  # Coil
    hr=ModbusSequentialDataBlock(0, [0]*100),  # Holding Register
    ir=None
)

context = ModbusServerContext(slaves=store, single=True)

# =========================
# 模擬PLC邏輯
# =========================
def plc_logic():
    print("PLC 模擬啟動...")
    running = False

    while True:
        # 讀 coils
        coils = context[0].getValues(1, 0, count=5)

        M200 = coils[0]  # Start
        M201 = coils[1]  # Stop
        M202 = coils[2]  # reset
        M203 = coils[3]
        M204 = coils[4]

        # 讀 registers
        regs = context[0].getValues(3, 0, count=5)

        # D200 (low, high)
        low = regs[0]
        high = regs[1]
        d200_val = (high << 16) | low

        # =====================
        # Start
        if M200:
            running = True
            print("PLC: Start")
            context[0].setValues(1, 1, [1])  # M201 = 完成

        # Stop
        if M201:
            running = False
            print("PLC: Stop")

        # Reset
        if M202:
            print("PLC: Reset Counter")
            context[0].setValues(3, 2, [0, 0])  # D202 = 0
            context[0].setValues(1, 4, [1])     # M204 = reset完成

        # =====================
        # 模擬計數回傳
        if running:
            # 把 PC送來的值寫到 D202
            context[0].setValues(3, 2, [low, high])

        time.sleep(0.5)

# =========================
# 啟動 Server
# =========================
if __name__ == "__main__":
    threading.Thread(target=plc_logic, daemon=True).start()

    print("Modbus TCP Slave 啟動 (Port 5020)...")
    StartTcpServer(context, address=("0.0.0.0", 5020))
'''