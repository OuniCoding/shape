# -*- mode: python ; coding: utf-8 -*-
import sys
from glob import glob

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

exts = ['json','jpg','png','dll','ini','txt','pdf','config','ico','pdb','exe','xml','db']

extra_files = []
for ext in exts:
    extra_files += [(f, '.') for f in glob(fr'.\\*.{ext}')]

data_file = [] + extra_files
print(data_file)

a = Analysis(
    ['modbusTCP_monitor.py'],
    pathex=[],
    binaries=[],
    datas=data_file,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5','tensorflow','pandas','jinja2','tensorflow-cpu'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    exclude_binaries=False,
    name='modbusTCP',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    console=True,
    upx=True,
    icon='.\\Modbus_TCP.png',
)
#coll = COLLECT(
#    exe,
#    a.binaries,
#    a.zipfiles,
#    a.datas,
#    strip=False,
#    upx=True,
#    upx_exclude=[],
#    name='modbusTCP'
#)