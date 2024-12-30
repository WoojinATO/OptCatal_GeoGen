function iCoFluentUnit = actfluentserver(orb,fluentAasFile)
    fluentkey=char(textread(fluentAasFile,'%s'));
    generic_fluent_object=orb.string_to_object(fluentkey);
    iCoFluentUnit=AAS_CORBA.ICoFluentUnitHelper.narrow(generic_fluent_object);
end