function  orb  = initialize_orb()
    args = javaArray('java.lang.String', 1);
    props=[];
    orb=org.omg.CORBA.ORB.init(args,props);
end