function topdf(fighandle,filename,varargin)

if length(varargin)~=1
  style = '-dpdf';
else
  style = varargin{1};
end


% check input arguments
if nargin==1, filename = fighandle; fighandle = get(0,'currentfigure'); end;
if ~ischar(filename), error('Needs a filename'); end;
if isempty(ishandle(fighandle)) || ~ishandle(fighandle),
    error('No current figure found, or invalid figure handle specified');
end;

% save current figure attributes
attr = get(fighandle);

% switch to custom page size
set(fighandle,'units','inches','paperunits','inches');
pp = get(fighandle,'position');
set(fighandle,'papersize',pp(3:4),'paperposition',[0 0 pp(3:4)]);

% print figure
print(fighandle,style,filename);
% print(fighandle,'-dpdf',filename);

% restore figure attributes
set(fighandle,'units',attr.Units);
set(fighandle,'paperunits',attr.PaperUnits);
set(fighandle,'papertype',attr.PaperType);
set(fighandle,'papersize',attr.PaperSize);
set(fighandle,'paperposition',attr.PaperPosition);

