function wbKey = actwbserver(wbAasFile)
    wbKey=char(AAS_CORBA.WbCollaborativeClient.connectToWb(wbAasFile));
end