from setuptools import setup

APP = ['main.py']
APP_NAME = 'Biophysics-app-1.0'
DATA_FILES = [
    ('data_output', []),  # You can add files later if needed
    ('docs', []),         # Optional documentation
    # Add static resources as needed (e.g., GIFs, images)
]

OPTIONS = {
    'argv_emulation': True,
    'packages': [
        'PyQt5',
        'numba',
        'matplotlib',
        'numpy',
        'scipy',
        'joblib',
        'intracellular_transport',
        'intracellular_transport.auxiliary_tools',
        'intracellular_transport.computational_tools',
        'intracellular_transport.gui_components',
        'intracellular_transport.job_queuing_system',
        'intracellular_transport.launch_functions',
        'intracellular_transport.system_configuration',
    ],
    'includes': ['queue', 'multiprocessing'],
    'excludes': ['tkinter'],  # Exclude unused GUI backends
    'plist': {
        'CFBundleName': APP_NAME,
        'CFBundleShortVersionString': '1.0',
        'CFBundleIdentifier': 'com.yourname.biophysicsapp',
        'CFBundleVersion': '1.0.0',
        'NSHighResolutionCapable': True
    },
    # 'iconfile': 'assets/app_icon.icns',  ← Add this line later when ready
}

setup(
    app=APP,
    name=APP_NAME,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
