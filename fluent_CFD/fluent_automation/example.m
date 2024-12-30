orb=initialize_orb;
load_ansys_aas;
iCoFluentUnit=actfluentserver(orb,'aaS_FluentId.txt');
iFluentTuiInterpreter=iCoFluentUnit.getSchemeControllerInstance();


display(sprintf ('FLUENT>%s','report system proc-stats'));
fluentResult=iFluentTuiInterpreter.doMenuCommandToString('report system proc-stats');
display(char(fluentResult));

display(sprintf ('FLUENT>%s','report summary -?'));
try
	fluentResult=iFluentTuiInterpreter.doMenuCommandToString('report summary -?');
	display(char(fluentResult));
catch exc
	if strcmp(exc.identifier,'MATLAB:Java:GenericException')==1
		if(strcmp(exc.ExceptionObject.getClass(),'class AAS_CORBA.EYesNoQuestion'))
			display(sprintf ('Fluent is asking a yes/no question:'));
			display(sprintf ('\t%s',char(exc.ExceptionObject.questionPromptWithDefaultAnswer)));
		end
	end
end

display(sprintf ('\tAnswering "no"..'));
try
	fluentResult=iFluentTuiInterpreter.doMenuCommandToString('report summary no -?');
	display(char(fluentResult));
catch exc
	if strcmp(exc.identifier,'MATLAB:Java:GenericException')==1
		if(strcmp(exc.ExceptionObject.getClass(),'class AAS_CORBA.EYesNoQuestion'))
			display(sprintf ('Fluent is asking a yes/no question:'));
			display(sprintf ('\t%s',char(exc.ExceptionObject.questionPromptWithDefaultAnswer)));
		end
	end
end