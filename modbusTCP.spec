# -*- mode: python ; coding: utf-8 -*-
import sys
from glob import glob

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

exts = ['jpg','dll','ini','txt','pdf','config','ico','pdb','exe','xml','db','json']

extra_files = []
for ext in exts:
    extra_files += [(f, '.') for f in glob(fr'..\*.{ext}')]

data_file = [
    ('..\\cfg','.\\cfg'),
    ('..\\data','.\\data'),
    ('..\\font','.\\font'),
    ('..\\Settings','.\\Settings'),
    ('..\\weights','.\\weights'),
    ('..\\MvImport','.\\MvImport'),
    ] + extra_files

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
    [],
    exclude_binaries=True,
    name='OUNI_mutil',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='.\\logo.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='modbusTCP_monitor',
)