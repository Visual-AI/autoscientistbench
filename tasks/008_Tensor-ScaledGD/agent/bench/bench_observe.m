function value = bench_observe(X, t, method, case_index)
  persistent input_pipe output_pipe bridge_pid;
  if isempty(input_pipe)
    bridge = fullfile(getenv('BENCH_BENCH_DIR'), 'octave_rpc.py');
    preload = getenv('LD_PRELOAD'); setenv('LD_PRELOAD', '');
    [input_pipe, output_pipe, bridge_pid] = popen2('python', {bridge});
    setenv('LD_PRELOAD', preload);
    if bridge_pid < 0; error('cannot start oracle bridge'); end
  end
  if strcmp(method,'finish_primary')
    request=struct('method',method);
  else
  filename = fullfile(tempdir(), sprintf('tensor-%d.bin', getpid()));
  fid = fopen(filename, 'wb', 'ieee-le');
  fwrite(fid, real(X(:)), 'double');
  is_complex = ~isreal(X);
  if is_complex; fwrite(fid, imag(X(:)), 'double'); end
  fclose(fid);
  request = struct('file', filename, 'shape', size(X), 'complex', is_complex, 'case', case_index, 'method', method, 'iteration', t);
  end
  fputs(input_pipe, [jsonencode(request), sprintf('\n')]); fflush(input_pipe);
  started = tic();
  while true
    line = fgetl(output_pipe);
    if ischar(line); break; end
    if toc(started) > 120; error('private oracle response timeout'); end
    fclear(output_pipe); pause(0.001);
  end
  if strcmp(line, 'ERROR'); error('private oracle rejected request'); end
  value = str2double(line);
  if ~isfinite(value); error('nonfinite oracle response'); end
end
