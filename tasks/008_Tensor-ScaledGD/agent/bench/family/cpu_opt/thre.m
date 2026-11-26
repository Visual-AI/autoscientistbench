function S = thre(S, theta)
S = sign(S) .* max(abs(S)-theta, 0.0);
end
