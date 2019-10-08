@echo off
python -c "if __name__ == '__main__': from neurodecode.utils.fif_info import batch_run; batch_run(r'%1')"
