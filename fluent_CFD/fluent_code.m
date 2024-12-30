%% add path
addpath(genpath('fluent_automation'))
addpath(genpath('sph'))

% 자동화 코드
jarPath=fileparts(which('initialize_orb'));
jarPathAndFile=sprintf('%s/%s',jarPath,'ANSYS_AAS.jar');
javaaddpath(jarPathAndFile);

args = javaArray('java.lang.String', 1);
props=[];
orb=org.omg.CORBA.ORB.init(args,props);


fluentAasFile = 'aaS_FluentId.txt';
fluentkey=char(textread(fluentAasFile, '%s'));
generic_fluent_object=orb.string_to_object(fluentkey);
iCoFluentUnit=AAS_CORBA.ICoFluentUnitHelper.narrow(generic_fluent_object);


data.fluent=iCoFluentUnit;
data.tui = data.fluent.getSchemeControllerInstance();



data.tui.doMenuCommand('rc inner_08_2.msh') %workbench에서 생성된 파일을 불러옴
data.tui.doMenuCommand('file/read-journal nox_journal_1')
data.tui.doMenuCommand('no')
data.tui.doMenuCommand('file/read-journal nox_journal_2')
data.tui.doMenuCommand('file/read-journal nox_journal_3')
data.tui.doMenuCommand('solve/initialize hyb-initialization yes')
data.tui.doMenuCommand('solve iterate 500')

temp_results=data.tui.doMenuCommandToString('/report/surface-integrals/integral/ inlet outlet () concentration-nh3 no');
temp_results=strsplit(string(temp_results));
inlet_X = str2double(temp_results(find(temp_results=="inlet")+1));
outlet_X = str2double(temp_results(find(temp_results=="outlet")+1));
conversion = (inlet_X-outlet_X)/(inlet_X);

temp_results_target=data.tui.doMenuCommandToString('/report/surface-integrals/integral/ inlet outlet () concentration-n2o no');
temp_results_target=strsplit(string(temp_results_target));
inlet_S = str2double(temp_results_target(find(temp_results_target=="inlet")+1));
outlet_S = str2double(temp_results_target(find(temp_results_target=="outlet")+1));
selectivity = (outlet_S-inlet_S)/(inlet_X-outlet_X);


data.tui.doMenuCommand('file/write-case-data final_175_3_inner_08_2')











