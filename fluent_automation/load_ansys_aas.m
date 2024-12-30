function  load_ansys_aas()
display('Start Loading AnsysAAS toolbox...')
matlab_version=version();
toolbox_version='1.1.6';
jarPath=fileparts(which('initialize_orb'));
jarPathAndFile=sprintf('%s/%s',jarPath,'ANSYS_AAS.jar');

javaaddpath(jarPathAndFile);

end