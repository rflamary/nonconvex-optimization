function options=initoptions(fname,options,optname)
% options=initoptions(fname,options)
% function that automatically load default options from comments 
% in a matlab function
% example of options parameters
%
% options parameters:
%   options.test1    Crappy value to set (default=10)
%   options.test2    Crappy value to also set (default='testdestring')

if nargin<2
    options=struct();
end


if nargin<3
    optname='options';
end


pattern_opt=[optname '\.(\w+)' ];
pattern_val=['\(default=(.+)\)' ];

fileID = fopen([fname '.m'],'r');

tline = fgetl(fileID);
while ischar(tline)
    if ~isempty(tline)
        if tline(1)=='%'
            opt=regexp(tline,pattern_opt,'tokens');
            if ~isempty(opt)
                val=regexp(tline,pattern_val,'tokens');
                if ~isempty(val)
                    tsk=['temp=' val{1}{1} ';'];
                    eval(tsk);
                    if ~isfield(options,opt{1}{1})
                        options.(opt{1}{1}) =temp;
                    end
                end
            end
            
        end
    end
    tline = fgetl(fileID);
end

fclose(fileID);
