@echo off
python -c "if __name__ == '__main__': import pycnbi.stream_recorder.stream_recorder as m; m.batch_run(r'%1', r'%2', r'%3')"
