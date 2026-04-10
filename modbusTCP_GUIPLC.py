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

# =========================
# 建立 PLC 資料區
# =========================
store = ModbusDeviceContext(
    di=ModbusSequentialDataBlock(0, [0]*100),
    co=ModbusSequentialDataBlock(0, [0]*100),
    hr=ModbusSequentialDataBlock(0, [0]*100),
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
        self.root.geometry("420x500")

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
        frame1 = ttk.LabelFrame(root, text="Coil 控制 (M200~M204)")
        frame1.pack(fill="x", padx=10, pady=5)

        self.coil_vars = []
        for i in range(5):
            var = tk.IntVar()
            chk = ttk.Checkbutton(
                frame1,
                text=f"M{200+i}",
                variable=var,
                command=lambda idx=i, v=var: self.set_coil(idx, v)
            )
            chk.grid(row=0, column=i, padx=5)
            self.coil_vars.append(var)

        # ===== Register 控制 =====
        frame2 = ttk.LabelFrame(root, text="D202/D203 寫入")
        frame2.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame2, text="32-bit值:").grid(row=0, column=0)
        self.reg_entry = ttk.Entry(frame2, width=15)
        self.reg_entry.grid(row=0, column=1)

        ttk.Button(frame2, text="寫入", command=self.write_register).grid(row=0, column=2, padx=10)

        # ===== 顯示區 =====
        frame3 = ttk.LabelFrame(root, text="狀態")
        frame3.pack(fill="both", expand=True, padx=10, pady=5)

        self.text = tk.Text(frame3)
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

        def run_server():
            StartTcpServer(context=context, address=("0.0.0.0", 5020))

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

    def write_register(self):
        try:
            val = int(self.reg_entry.get())

            low = val & 0xFFFF
            high = (val >> 16) & 0xFFFF

            context[0].setValues(3, 2, [low, high])

        except:
            pass

    # ------------------------
    def update_view(self):
        while True:
            try:
                coils = context[0].getValues(1, 0, count=5)
                if coils[1] == 1:    # Stop
                    context[0].setValues(3, 0, [0, 0])
                    context[0].setValues(3, 2, [0, 0])
                regs = context[0].getValues(3, 0, count=5)
                # d200_201 = context[0].getValues(3, 0, count=2)
                # d202_203 = context[0].getValues(3, 2, count=2)
                # d204 = context[0].getValues(3, 3, count=1)

                text = []

                text.append("\n=== Register ===")
                for i, v in enumerate(regs):
                    text.append(f"D{200+i} = {v}")

                text.append("=== Coil ===")
                for i, v in enumerate(coils):
                    text.append(f"M{200+i} = {v}")

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