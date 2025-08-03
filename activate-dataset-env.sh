#!/bin/bash
# Activation script for the minimal dataset creation environment

echo "🔧 Activating dataset creation environment..."
source venv-dataset/bin/activate

echo "✅ Virtual environment activated!"
echo "📦 Installed packages:"
pip list

echo ""
echo "🚀 Ready to create OpenAssistant dataset!"
echo "Usage: python dataset/build_openassistant_dataset.py [options]"
echo ""
echo "📖 For help: python dataset/build_openassistant_dataset.py --help"
echo "💡 Example: python dataset/build_openassistant_dataset.py --subsample-size 1000 --max-seq-len 256"