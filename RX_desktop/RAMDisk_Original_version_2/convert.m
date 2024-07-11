function convert(vectorIn, name)
N = length(vectorIn);

vec = zeros(2*N,1);

% Normalize
vectorIn = vectorIn / max(abs(vectorIn)) * 0.7;

I = real(vectorIn);
Q = imag(vectorIn);

% Filling the vector with previously generated I and Q
j = 1;
for i=1:N
    vec(j) = I(i);
    j=j+1;
    vec(j) = Q(i);
    j=j+1;
end

% Printing to file
FID = fopen(name,'w+');
fprintf(FID,'%f\r\n',vec);
fclose(FID);


end