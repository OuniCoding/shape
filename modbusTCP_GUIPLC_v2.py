import tkinter as tk
from tkinter import ttk
import threading
import time

from pymodbus.server import StartTcpServer
from pymodbus.server import ModbusTcpServer
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusDeviceContext,
    ModbusServerContext
)

# ============================================================
# 自訂 PLC D Register
#
# 保留原本的：
#   Modbus address 0  -> D200
#   Modbus address 1  -> D201
#   Modbus address 2  -> D202
#   Modbus address 3  -> D203
#   Modbus address 4  -> D204
#
# 新增：
#   Modbus address 20000 -> D20000
#   Modbus address 20001 -> D20001
#   ...
#   Modbus address 20024 -> D20024
#
# 因此不需要把 D20000 換算成 19800
# ============================================================
class PLCRegisterBlock(ModbusSequentialDataBlock):

    def __init__(self):

        # 先建立原本的資料區
        self.data = {
            201: 0,   # D200
            202: 0,   # D201
            203: 0,   # D202
            204: 0,   # D203
            205: 0,   # D204
        }

        # 額外建立 D20000 ~ D20024
        # D20000 ~ D20024
        for address in range(20001, 20026):     # for address in range(20000, 20025):
            self.data[address] = 0

    # --------------------------------------------------------
    # 判斷 Modbus address 是否存在
    # --------------------------------------------------------
    def validate(self, address, count=1):

        # 原本 D200 ~ D204
        if 0 <= address and address + count <= 5:
            return True

        # D20000 ~ D20024
        if 20000 <= address and address + count <= 20025:
            return True

        return False
    # --------------------------------------------------------
    # 讀取 Register
    # --------------------------------------------------------
    def getValues(self, address, count=1):
        result = []
        for i in range(count):

            addr = address + i

            if addr in self.data:
                result.append(self.data[addr])  # self.data[addr]
            else:
                # pymodbus 內部可能會使用 +1 位址
                # 因此再嘗試 -1
                real_addr = addr - 1

                if real_addr in self.data:
                    result.append(self.data[real_addr])
                else:
                    return []

        return result

    # --------------------------------------------------------
    # 寫入 Register
    # --------------------------------------------------------
    def setValues(self, address, values):
        for i, value in enumerate(values):
            addr = address + i
            if addr in self.data:
                self.data[addr] = value & 0xFFFF
            elif (addr - 1) in self.data:
                self.data[addr - 1] = value & 0xFFFF
            else:
                raise ValueError(
                    f"不支援的 PLC Register Address: {addr}"
                )
    # ========================================================
    # PLC實際位址讀取
    # ========================================================
    def get_D(self, address):
        if address in self.data:
            return self.data[address]
        raise ValueError(
            f"D{address} 不存在"
        )

    # ========================================================
    # PLC實際位址寫入
    # ========================================================
    def set_D(self, address, value):
        if address not in self.data:
            raise ValueError(
                f"D{address} 不存在"
            )
        self.data[address] = value & 0xFFFF

# =========================
# 建立 PLC 資料區
# =========================

hr_block = PLCRegisterBlock()

store = ModbusDeviceContext(
    di=ModbusSequentialDataBlock(0, [0]*100),
    co=ModbusSequentialDataBlock(0, [0]*100),
    hr=hr_block,    # hr=ModbusSequentialDataBlock(0, [0]*100),
    ir=ModbusSequentialDataBlock(0, [0]*100),
)

context = ModbusServerContext(devices=store, single=True)

server_running = False
server_thread = None
server = None

# =========================
# GUI
# =========================
class PLC_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PLC 模擬器 (Modbus TCP 3.x)")
        self.root.geometry("600x900")   # ("420x500")

        # ===== Server 控制 =====
        frame0 = ttk.LabelFrame(root, text="Server 控制")
        frame0.pack(fill="x", padx=10, pady=5)

        style = ttk.Style()
        style.configure('Red.TButton', foreground='red')  # 設定名為 Red.TButton 的樣式，字體紅色
        style.configure('Green.TButton', foreground='green')  # 設定名為 Green.TButton 的樣式，字體綠色
        style.configure('Blue.TButton', foreground='blue')  # 設定名為 Blue.TButton 的樣式，字體藍色

        ttk.Button(frame0, text="▶ Start", command=self.start_server, style='Green.TButton').grid(row=0, column=0, padx=10)
        ttk.Button(frame0, text="⏹ Stop", command=self.stop_server, style='Red.TButton').grid(row=0, column=1, padx=10)

        self.status = ttk.Label(frame0, text="未啟動", foreground="red")
        self.status.grid(row=0, column=2, padx=10)

        # ===== Coil 控制 =====
        frame1 = ttk.LabelFrame(root, text="Coil 控制 (M200~M213)")
        frame1.pack(fill="x", padx=10, pady=5)

        self.coil_vars = {}
        for i in range(14):
            if (i >= 3 and i <= 9):
                continue
            var = tk.IntVar()
            chk = ttk.Checkbutton(
                frame1,
                text=f"M{200+i}",
                variable=var,
                command=lambda idx=i, v=var: self.set_coil(idx, v)
            )
            chk.grid(row=0, column=i, padx=5)
            # self.coil_vars.append(var)
            self.coil_vars[i] = var

        # ===== Register 控制 =====
        frame2 = ttk.LabelFrame(root, text="D202/D203 寫入")
        frame2.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame2, text="32-bit值:").grid(row=0, column=0, padx=5)    # .grid(row=0, column=0)
        self.reg_entry = ttk.Entry(frame2, width=18)    # ttk.Entry(frame2, width=15)
        self.reg_entry.grid(row=0, column=1, padx=5)    # grid(row=0, column=1)

        ttk.Button(frame2, text="寫入", command=self.write_register).grid(row=0, column=2, padx=10)

        # ====================================================
        # D20000 ~ D20024 手動輸入
        # ====================================================

        frame4 = ttk.LabelFrame(root, text="D20000 ~ D20024 手動寫入")
        frame4.pack(fill="x", padx=10,  pady=5)

        ttk.Label(frame4, text="D位址:").grid(row=0, column=0, padx=5)

        self.d_address_entry = ttk.Entry(frame4, width=10)
        self.d_address_entry.insert(0, "20000")
        self.d_address_entry.grid(row=0, column=1, padx=5)

        ttk.Label(frame4, text="數值:").grid( row=0, column=2, padx=5)

        self.d_value_entry = ttk.Entry(frame4, width=12)
        self.d_value_entry.grid(row=0, column=3, padx=5)

        ttk.Button(frame4, text="寫入", command=self.write_d20000).grid(row=0, column=4, padx=10)

        # ===== 顯示區 =====
        frame3 = ttk.LabelFrame(root, text="狀態")
        frame3.pack(fill="both", expand=True, padx=10, pady=5)

        # self.text = tk.Text(frame3)
        self.text = tk.Text(frame3, font=("Consolas", 10) )
        self.text.pack(fill="both", expand=True)

        # 更新執行緒
        threading.Thread(target=self.update_view, daemon=True).start()

    # ------------------------

    def start_server(self):
        global server_running, server_thread, server

        if server_running:
            return

        server_running = True
        self.status.config(text="運行中", foreground="green")

        for address in range(20001, 20026):
            hr_block.set_D(address, 0)  # clean D20000~D20024

        def run_server():
            global server
            try:
                StartTcpServer(context=context, address=("0.0.0.0", 5020))
            except Exception as e:
                print("Modbus Server Error:", e)

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        #     try:
        #         StartTcpServer(context=context, address=("0.0.0.0", 5020))
        #     except:
        #         pass
        # server_thread = threading.Thread(target=run, daemon=True)
        # server_thread.start()

    def stop_server(self):
        global server_running, server

        if not server_running:
            return

        server_running = False
        self.status.config(text="已停止", foreground="red")

        try:
            if server:
                server.shutdown()  # 停止 serve_forever

                server = None
        except Exception as e:
            print("Stop Error:", e)

#         server_running = False
#         self.status.config(text="已停止", foreground="red")

#         # ⚠️ pymodbus 沒有優雅關閉 → 用強制方式模擬斷線
#         import os
#         os._exit(0)


    # ------------------------
    def set_coil(self, index, var):
        val = var.get()
        context[0].setValues(1, index, [val])

    # ========================================================
    # D202 / D203
    #
    # 32-bit
    #
    # D202 = Low
    # D203 = High
    # ========================================================
    def write_register(self):
        try:
            val = int(self.reg_entry.get())

            low = val & 0xFFFF
            high = (val >> 16) & 0xFFFF

            # context[0].setValues(3, 2, [low, high])

            hr_block.set_D(203, low)    # 202
            hr_block.set_D(204, high)   # 203

        except Exception as e:
            print("D202/D203 寫入錯誤:", e)

    # ========================================================
    # D20000 ~ D20024 手動寫入
    # ========================================================
    def write_d20000(self):
        try:
            address = int(self.d_address_entry.get()) + 1
            value = int(self.d_value_entry.get())

            # if not (20000 <= address <= 20024):
            if not (20001 <= address <= 20026):
                raise ValueError("D位址必須在 D20000 ~ D20024")

            # hr_block.setValues(address,[value])
            hr_block.set_D(address, value)
            print(f"D{address} = {value}")
        except Exception as e:
            print("D20000 寫入錯誤:", e)

    # ------------------------
    def update_view(self):
        while True:
            try:
                coils = context[0].getValues(1, 0, count=14)
                # ====================================================
                # Modbus Coil → GUI Checkbutton 同步
                # ====================================================

                if coils[1] == 1:    # Stop
                    # context[0].setValues(3, 0, [0, 0])
                    # context[0].setValues(3, 2, [0, 0])
                    hr_block.set_D(201, 0)  # D200
                    hr_block.set_D(202, 0)  # D201
                    hr_block.set_D(203, 0)  # D202
                    hr_block.set_D(204, 0)  # D203

                # ============================================
                # 原本 D200 ~ D204
                # ============================================
                regs = [
                    hr_block.get_D(201),          # D200
                    hr_block.get_D(202),          # D201
                    hr_block.get_D(203),          # D202
                    hr_block.get_D(204),          # D203
                    hr_block.get_D(205),          # D204
                ]
                # ============================================
                # D20000 ~ D20024
                # ============================================
                d20000_regs = [
                    hr_block.get_D(address)
                    # for address in range(20000, 20025)
                    for address in range(20001, 20026)
                ]

                text = []

                # ============================================
                # D200 ~ D204
                # ============================================
                text.append("\n=== Register D200 ~ D204 ===")
                for i, v in enumerate(regs):
                    text.append(f"D{200+i} = {v}")

                # ============================================
                # D20000 ~ D20024
                # ============================================
                text.append("\n=== Register D20000 ~ D20024 ===")
                for i, v in enumerate(d20000_regs):
                    text.append(f"D{20000+i:<5} = {v}")

                text.append("=== Coil M200 ~ M213 ===")
                for i, v in enumerate(coils):
                    if (i >= 3 and i <= 9):
                        continue
                    text.append(f"M{200+i} = {v}")
                    self.coil_vars[i].set(1 if coils[i] else 0)

                self.text.delete(1.0, tk.END)
                self.text.insert(tk.END, "\n".join(text))

            except Exception as e:
                self.text.insert(tk.END, f"錯誤: {e}")

            time.sleep(0.5)

# =========================
# 主程式
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = PLC_GUI(root)
    root.mainloop()