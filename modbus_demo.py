"""
-------------------------------------------------------------------------------
對應 Arduino 暫存器表
類型	位址	說明	Python操作
Coil	0	啟動（Str）	write_coil(0, True)
Coil	1	停止（Stop）	write_coil(1, True)
Holding Register	0	Go 數值	write_register(2, val)
Holding Register	1	targetCount (Ct)	write_register(3, val)
Holding Register	2	triggerCount (Cs)	write_register(4, val)
Holding Register	5	TriggerCount 回報	read_holding_registers(5,1)
Holding Register	6	BufIndex 回報	read_holding_registers(6,1)
Holding Register	7	OUT Flag(en_out_flag)	read_holding_registers(7,1)
Holding Register	8	set 計時器設定值 (Timer Value)	read_holding_registers(8,1)
Holding Register	9	get 計時器設定值 (Timer Value)	read_holding_registers(9,1)
--------------------------------------------------------------------------------
pip install pymodbus pyserial
"""
from pymodbus.client import ModbusSerialClient as ModbusClient
import time

client = ModbusClient(
    # method='rtu',
    port='COM3',        # 請改成你的 RS232 COM 埠
    baudrate=115200,
    parity='N',
    stopbits=1,
    bytesize=8,
    timeout=1
)

if not client.connect():
    print("❌ 無法連線到 Modbus 裝置")
    exit()

print("✅ 已連線到 Arduino (Slave ID = 1)")

SLAVE_ID = 1

# 寫入 Coil 啟動命令
client.write_coil(0, True, device_id=SLAVE_ID)  # Start
print("→ 啟動信號已送出")

client.write_register(0, 5, device_id=SLAVE_ID)
print("→ Go信號已送出5")

# 讀取 Holding Register 狀態
time.sleep(0.5)
while True:
    # result = client.read_coils(0, count=1, device_id=SLAVE_ID)
    result = client.read_holding_registers(7, count=1, device_id=SLAVE_ID)
    if result.isError():
        print("❌ 讀取失敗")
    else:
        print("OUT Flag 狀態暫存器資料:", result.registers)
    result = client.read_holding_registers(5, count=1, device_id=SLAVE_ID)
    if result.isError():
        print("❌ 讀取失敗")
    else:
        print("TriggerCount暫存器資料:", result.registers)
        if result.registers == [20]:
            client.write_coil(1, True, device_id=SLAVE_ID)  # Stop
            break

    result = client.read_holding_registers(0, count=10, device_id=SLAVE_ID)
    if result.isError():
        print("❌ 讀取失敗")
    else:
        print("BufIndex暫存器資料:", result.registers)
    #time.sleep(1)

client.close()
