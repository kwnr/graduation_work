pe=pyenv;
if pe.Status == 'NotLoaded'
    pyenv("ExecutionMode","OutOfProcess","Version",'/usr/local/opt/python@3.10/bin/python3.10');
end

F.f=figure;
F.btn=uicontrol("Style","pushbutton",'Units','pixels','Position',[1000,800,75,25],'String','close','Callback','delete(gcbf)');
cam=webcamlist(1);



