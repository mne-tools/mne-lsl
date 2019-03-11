import pycnbi.stream_player.stream_player as sp

if __name__ == '__main__':
    server_name = 'StreamPlayer'
    chunk_size = 4  # chunk streaming size
    fif_file = r'D:\data\STIMO_EEG\DM002\offline\all\20180620-115808-raw.fif'
    trigger_file = 'triggerdef_gait_chuv.ini'
    sp.stream_player(server_name, fif_file, chunk_size, verbose='events', trigger_file=trigger_file)
