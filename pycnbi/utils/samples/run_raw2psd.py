from __future__ import print_function, division

"""
Example code for computing PSD features over a sliding window

Kyuhwa Lee
Swiss Federal Institute of Technology (EPFL)

"""

if __name__ == '__main__':
    import pycnbi
    import pycnbi.utils.q_common as qc
    from raw2psd import raw2psd

    data_dir = r'D:\data\MI\rx1\offline\gait-pulling\20161104'
    n_jobs = 1
    fmin = 1
    fmax = 40
    wlen = 0.5
    wstep = 32
    tmin = 0.0
    tmax = 30
    channel_picks = None
    excludes = ['TRIGGER', 'M1', 'M2', 'EOG']

    if n_jobs > 1:
        import multiprocessing as mp

        pool = mp.Pool(n_jobs)
        procs = []
        for rawfile in qc.get_file_list(data_dir):
            if rawfile[-8:] != '-raw.fif': continue
            cmd = [rawfile, fmin, fmax, wlen, wstep, tmin, tmax, channel_picks, excludes]
            procs.append(pool.apply_async(raw2psd, cmd))
        for proc in procs:
            proc.get()
        pool.close()
        pool.join()
    else:
        for rawfile in qc.get_file_list(data_dir):
            if rawfile[-8:] != '-raw.fif': continue
            raw2psd(rawfile, fmin=fmin, fmax=fmax, wlen=wlen, wstep=wstep, tmin=tmin,
                    tmax=tmax, channel_picks=channel_picks, excludes=excludes)

    print('Done.')
