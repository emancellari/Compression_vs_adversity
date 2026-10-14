"""Analyze the supplied NbuGAN-0.90 clean/adversarial pair using JPEG."""
# Use any clean/adversarial pair from any attack with --clean and --adversarial.
# Use any classifier/dataset with --adapter module:factory; see compression_analysis.py.
from compression_analysis import main

if __name__ == '__main__':
    main('jpeg')
