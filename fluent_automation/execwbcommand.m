function response = execwbcommand(wbCommand)
    response=char(AAS_CORBA.WbCollaborativeClient.executewbCommand(wbCommand));
end