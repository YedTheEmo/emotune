"""
Warning management utilities for EmoTune
"""

import os
import warnings
import logging

def suppress_tensorflow_warnings():
    """Suppress TensorFlow oneDNN warnings"""
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
    
    # Suppress specific warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
    warnings.filterwarnings('ignore', message='.*oneDNN.*')

def suppress_pyo_warnings():
    """Suppress non-critical Pyo warnings"""
    # Suppress WxPython warnings (these are just informational)
    warnings.filterwarnings('ignore', message='.*WxPython.*')
    warnings.filterwarnings('ignore', message='.*Portmidi closed.*')
    warnings.filterwarnings('ignore', message='.*Pyo warning.*')
    warnings.filterwarnings('ignore', message='.*Pyo error.*')
    
    # Suppress print statements from Pyo
    import builtins
    original_print = builtins.print
    
    def filtered_print(*args, **kwargs):
        message = ' '.join(str(arg) for arg in args)
        if any(keyword in message.lower() for keyword in ['wxpython', 'portmidi closed', 'pyo warning', 'pyo error']):
            return
        original_print(*args, **kwargs)
    
    builtins.print = filtered_print

def configure_logging_for_tests():
    """Configure logging to reduce noise during tests"""
    # Set logging level to reduce noise
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('pyo').setLevel(logging.WARNING)
    
    # Suppress specific noisy loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

def setup_environment():
    """Setup environment for clean operation"""
    suppress_tensorflow_warnings()
    suppress_pyo_warnings()
    configure_logging_for_tests() 