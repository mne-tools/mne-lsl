@echo off
python -c "if __name__ == '__main__': from neurodecode.utils.fif_resample import batch_run; batch_run(r'%1', '%2')"
