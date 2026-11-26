function bench_setup()
  warning('off', 'Octave:shadowed-function');
  pkg('local_list', fullfile(getenv('BENCH_RUNTIME_DIR'), 'octave-config', 'octave_packages'));
  pkg load optim
  pkg load statistics
  addpath(fullfile(fileparts(mfilename('fullpath')), 'code (Octave)'));
end
