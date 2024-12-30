function iCoMapdlUnit = actmapdlserver(orb,mapdlAasFile)
    mapdlkey=char(textread(mapdlAasFile,'%s'));
    generic_mapdl_object=orb.string_to_object(mapdlkey);
    iCoMapdlUnit=AAS_CORBA.MAPDL.Solvers.ICoMapdlUnitHelper.narrow(generic_mapdl_object);
end