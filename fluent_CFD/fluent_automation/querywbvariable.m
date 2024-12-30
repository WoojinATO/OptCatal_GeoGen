function response = querywbvariable(wbVariable)
    response=char(AAS_CORBA.WbCollaborativeClient.retrieveWbVariable(wbVariable));
end