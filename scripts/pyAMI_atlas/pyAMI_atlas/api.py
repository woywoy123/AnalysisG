# -*- coding: utf-8 -*-
from __future__ import (division, print_function, unicode_literals)
#############################################################################
# Author  : Jerome ODIER, Jerome FULACHIER, Fabian LAMBERT, Solveig ALBRAND
#
# Email   : jerome.odier@lpsc.in2p3.fr
#           jerome.fulachier@lpsc.in2p3.fr
#           fabian.lambert@lpsc.in2p3.fr
#           solveig.albrand@lpsc.in2p3.fr
#
# Version : 5.X.X (2014)
#
#############################################################################

import re, json, pyAMI.utils, pyAMI.exception, pyAMI_atlas.utils

#############################################################################

def init():
	'''Initialize the ATLAS specific API. !!! DEPRECATED !!!

	Returns:
	    nothing.
	'''

	pass

#############################################################################
# SHOW.AMI-TAG                                                              #
#############################################################################

def get_ami_tag(client, tag):
	'''Get AMI-Tag information.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]
	    :tag: the AMI-Tag [ str ]

	Returns:
	    an array of python dictionnaries.
	'''

	command = [
		'AMIGetAMITagInfo',
		'-amiTag="%s"' % tag,
	]

	return client.execute(command, format = 'dom_object').get_rows('amiTagInfo')

#############################################################################

def mode_show_ami_tag(client, args):
	amiTags = get_ami_tag(client, args['tag'])

	for amiTag in amiTags:
		amiTag['tagName'] = amiTag['tagType'] + amiTag['tagNumber']

	pyAMI_atlas.utils.print_list(client, amiTags, title = 'tagName')

	return 0

#############################################################################
# SHOW.HIERARCHICAL.AMI-TAG                                                 #
#############################################################################

#def get_hierarchical_ami_tag(client, tag):
#	'''Get AMI-Tag information.
#
#	Args:
#	    :client: the pyAMI client [ pyAMI.client.Client ]
#	    :tag: the AMI-Tag [ str ]
#
#	Returns:
#	    an array of python dictionnaries.
#	'''
#
#	command = [
#		'AMIGetAMITagInfo',
#		'-amiTag="%s"' % tag,
#		'-hierarchicalView'
#	]
#
#	return [json.loads(amiTag['dict']) for amiTag in client.execute(command, format = 'dom_object').get_rows('amiTagInfo')]

#############################################################################

#def mode_show_hierarchical_ami_tag(client, args):
#	pyAMI.utils.print_json(get_hierarchical_ami_tag(client, args['tag']))
#
#	return 0

#############################################################################

#def add_hierarchical_ami_tag(client, python_dict):
#	'''Add a new hierarchical AMI-Tag.
#
#	Args:
#	    :client: the pyAMI client [ pyAMI.client.Client ]
#	    :python_dict: the hierarchical AMI-Tag [ dict ]
#
#	Returns:
#	    an array of python dictionnaries.
#	'''
#
#	command = [
#		'AddHierarchicalAMITag',
#		'-dict="%s"' % json.dumps(python_dict).replace('"', '\\"')
#	]
#
#	return client.execute(command, format = 'dom_object').get_rows('amiTag')

#############################################################################

#def clone_hierarchical_ami_tag(client, python_dict):
#	'''Clone a hierarchical AMI-Tag.
#
#	Args:
#	    :client: the pyAMI client [ pyAMI.client.Client ]
#	    :tag: the AMI-Tag [ str ]
#	    :python_dict: the hierarchical AMI-Tag [ dict ]
#
#	Returns:
#	    an array of python dictionnaries.
#	'''
#
#	command = [
#		'CloneHierarchicalAMITag',
#		'-amiTag="%s"' % tag,
#		'-dict="%s"' % json.dumps(python_dict).replace('"', '\\"')
#	]
#
#	return client.execute(command, format = 'dom_object').get_rows('amiTag')

#############################################################################
# SHOW.DATASET.INFO                                                         #
#############################################################################

def get_dataset_info(client, dataset):
	'''Get dataset information.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]
	    :dataset: the dataset [ str ]

	Returns:
	    a python dictionnaries.
	'''

	command = [
		'AMIGetDatasetInfo',
		'-logicalDatasetName="%s"' % dataset,
	]

	return client.execute(command, format = 'dom_object').get_rows()

#############################################################################

def mode_show_dataset_info(client, args):
	pyAMI_atlas.utils.print_list(client, get_dataset_info(client, args['dataset']))

	return 0

#############################################################################
# SHOW.DATASET.HASHTAGS                                                     #
#############################################################################

def get_dataset_hashtags(client, dataset):
	'''Get dataset information.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]
	    :dataset: the dataset [ str ]

	Returns:
	    a python dictionnaries.
	'''

	command = [
		'DatasetWBListHashtags',
		'-ldn="%s"' % dataset,
	]

	return client.execute(command, format = 'dom_object').get_rows()

#############################################################################
# SHOW.DATASET.PROV                                                         #
#############################################################################

def get_dataset_prov(client, dataset):
	'''Get dataset provenance.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]
	    :dataset: the dataset [ str ]

	Returns:
	    a map of python dictionnaries. The key "node" gives a list of dataset with the distance to the given dataset and the key "edge" gives the list of successive pairs of input an output datasets.
	'''

	command = [
		'AMIGetDatasetProv',
		'-logicalDatasetName="%s"' % dataset,
	]

	result = client.execute(command, format = 'dom_object')

	return {'node': result.get_rows('node'),
	          'edge': result.get_rows('edge')}

#############################################################################

def mode_show_dataset_prov(client, args):
	provenance = get_dataset_prov(client, args['dataset'])

	pyAMI_atlas.utils.print_table(client, provenance['node'])
	pyAMI_atlas.utils.print_table(client, provenance['edge'])

	return 0

#############################################################################

def mode_show_dataset_hashtags(client, args):
	pyAMI_atlas.utils.print_table(client, get_dataset_hashtags(client, args['dataset']))

	return 0

#############################################################################
# SHOW.FILE                                                                 #
#############################################################################

GUID_RE = re.compile('^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')

#############################################################################

def get_file(client, file):
	'''Get file information.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]
	    :file: the GUID or LFN [ str ]

	Returns:
	    a python dictionnaries.
	'''

	command = [
		'AMIGetFileInfo',
	]

	if GUID_RE.match(file):
		command.append('-GUID="%s"' % file)
	else:
		command.append('-LFN="%s"' % file)

	return client.execute(command, format = 'dom_object').get_rows()

#############################################################################

def mode_show_file(client, args):

	pyAMI_atlas.utils.print_list(client, get_file(client, args['file']))

	return 0

#############################################################################
# SHOW.PAPER                                                                #
#############################################################################

def get_paper(client, refCode):
	'''Get paper information.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]
	    :refCode: the Glance Reference Code [ str ]

	Returns:
	    a map of python dictionnaries. The key "paper" gives the paper info and the key "datasets" gives the list associated datasets.
	'''

	command = [
		'AMIGlanceGetPaperInfo',
		'-refCode="%s"' % refCode,
	]

	result = client.execute(command, format = 'dom_object')

	return {'paper': result.get_rows('paper'),
	          'datasets': result.get_rows('datasets')}

#############################################################################

def mode_show_paper(client, args):
	paper = get_paper(client, args['refCode'])

	pyAMI_atlas.utils.print_list(client, paper['paper'], title = 'refCode')

	pyAMI_atlas.utils.print_table(client, paper['datasets'])

	return 0

#############################################################################
# LIST COMMANDS                                                             #
#############################################################################

def list_commands(client):
	'''List AMI commands.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]

	Returns:
	    an array of python dictionnaries.
	'''

	command = [
		'AMIListCommands',
	]

	return client.execute(command, format = 'dom_object').get_rows()

#############################################################################

def mode_list_commands(client, args):
	pyAMI_atlas.utils.print_table(client, list_commands(client))

	return 0

#############################################################################
# LIST PROJECTS                                                             #
#############################################################################

def list_projects(client, patterns = None, fields = None, order = None, limit = None, show_archived = False, **kwargs):
	'''List ATLAS projects.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]
	    :patterns: the list of project patterns (glob with %) [ list<str> | None ]
	    :fields: the list of extra fields to be shown if the result [ list<str> | None ], type ``ami list projects --help`` in the shell to have the list of available fields
	    :order: the list of fields for ordering the result [ list<str> | None ], type ``ami list projects --help`` in the shell to have the list of available fields
	    :limit: limit number of results [ int | tuple<int,int> | None ]
	    :show_archived: show archived results [ bool ]

	kwargs:
	    additionnal constraints field_name:field_value (glob with %), type ``ami list projects --help`` in the shell to have the list of available fields

	Returns:
	    an array of python dictionnaries.
	'''

	if not 'read_status' in kwargs:
		kwargs['read_status'] = 'VALID'

	return pyAMI.utils.smart_execute(client, 'projects', patterns, fields, order, limit, show_archived, **kwargs).get_rows()

#############################################################################

def mode_list_projects(client, args):
	pyAMI_atlas.utils.print_table(client, list_projects(client, **args))

	return 0

#############################################################################
# LIST SUBPROJECTS                                                          #
#############################################################################

def list_subprojects(client, patterns = None, fields = None, order = None, limit = None, show_archived = False, **kwargs):
	'''List ATLAS subprojects.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]
	    :patterns: the list of subproject patterns (glob with %) [ list<str> | None ]
	    :fields: the list of extra fields to be shown if the result [ list<str> | None ], type ``ami list subprojects --help`` in the shell to have the list of available fields
	    :order: the list of fields for ordering the result [ list<str> | None ], type ``ami list subprojects --help`` in the shell to have the list of available fields
	    :limit: limit number of results [ int | tuple<int,int> | None ]
	    :show_archived: show archived results [ bool ]

	kwargs:
	    additionnal constraints field_name:field_value (glob with %), type ``ami list subprojects --help`` in the shell to have the list of available fields

	Returns:
	    an array of python dictionnaries.
	'''

	if not 'read_status' in kwargs:
		kwargs['read_status'] = 'VALID'

	return pyAMI.utils.smart_execute(client, 'subprojects', patterns, fields, order, limit, show_archived, **kwargs).get_rows()

#############################################################################

def mode_list_subprojects(client, args):
	pyAMI_atlas.utils.print_table(client, list_subprojects(client, **args))

	return 0

#############################################################################
# LIST TYPES                                                                #
#############################################################################

def list_types(client, patterns = None, fields = None, order = None, limit = None, show_archived = False, **kwargs):
	'''List ATLAS types.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]
	    :patterns: the list of type patterns (glob with %) [ list<str> | None ]
	    :fields: the list of extra fields to be shown if the result [ list<str> | None ], type ``ami list types --help`` in the shell to have the list of available fields
	    :order: the list of fields for ordering the result [ list<str> | None ], type ``ami list types --help`` in the shell to have the list of available fields
	    :limit: limit number of results [ int | tuple<int,int> | None ]
	    :show_archived: show archived results [ bool ]

	kwargs:
	    additionnal constraints field_name:field_value (glob with %), type ``ami list types --help`` in the shell to have the list of available fields

	Returns:
	    an array of python dictionnaries.
	'''

	if not 'read_status' in kwargs:
		kwargs['read_status'] = 'VALID'

	return pyAMI.utils.smart_execute(client, 'types', patterns, fields, order, limit, show_archived, **kwargs).get_rows()

#############################################################################

def mode_list_types(client, args):
	pyAMI_atlas.utils.print_table(client, list_types(client, **args))

	return 0

#############################################################################
# LIST SUBTYPES                                                             #
#############################################################################

def list_subtypes(client, patterns = None, fields = None, order = None, limit = None, show_archived = False, **kwargs):
	'''List ATLAS subtypes.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]
	    :patterns: the list of subtype patterns (glob with %) [ list<str> | None ]
	    :fields: the list of extra fields to be shown if the result [ list<str> | None ], type ``ami list subprojects --help`` in the shell to have the list of available fields
	    :order: the list of fields for ordering the result [ list<str> | None ], type ``ami list subprojects --help`` in the shell to have the list of available fields
	    :limit: limit number of results [ int | tuple<int,int> | None ]
	    :show_archived: show archived results [ bool ]

	kwargs:
	    additionnal constraints field_name:field_value (glob with %), type ``ami list subprojects --help`` in the shell to have the list of available fields

	Returns:
	    an array of python dictionnaries.
	'''

	if not 'read_status' in kwargs:
		kwargs['read_status'] = 'VALID'

	return pyAMI.utils.smart_execute(client, 'subtypes', patterns, fields, order, limit, show_archived, **kwargs).get_rows()

#############################################################################

def mode_list_subtypes(client, args):
	pyAMI_atlas.utils.print_table(client, list_subtypes(client, **args))

	return 0

#############################################################################
# LIST PRODSTEPS                                                            #
#############################################################################

def list_prodsteps(client, patterns = None, fields = None, order = None, limit = None, show_archived = False, **kwargs):
	'''List ATLAS prodsteps.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]
	    :patterns: the list of prodstep patterns (glob with %) [ list<str> | None ]
	    :fields: the list of extra fields to be shown if the result [ list<str> | None ], type ``ami list prodsteps --help`` in the shell to have the list of available fields
	    :order: the list of fields for ordering the result [ list<str> | None ], type ``ami list prodsteps --help`` in the shell to have the list of available fields
	    :limit: limit number of results [ int | tuple<int,int> | None ]
	    :show_archived: show archived results [ bool ]

	kwargs:
	    additionnal constraints field_name:field_value (glob with %), type ``ami list prodsteps --help`` in the shell to have the list of available fields

	Returns:
	    an array of python dictionnaries.
	'''

	if not 'read_status' in kwargs:
		kwargs['read_status'] = 'VALID'

	return pyAMI.utils.smart_execute(client, 'prodsteps', patterns, fields, order, limit, show_archived, **kwargs).get_rows()

#############################################################################

def mode_list_prodsteps(client, args):
	pyAMI_atlas.utils.print_table(client, list_prodsteps(client, **args))

	return 0

#############################################################################
# LIST NOMENCLATURE                                                         #
#############################################################################

def list_nomenclature(client, patterns = None, fields = None, order = None, limit = None, show_archived = False, **kwargs):
	'''List ATLAS nomenclature.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]
	    :patterns: the list of nomenclature patterns (glob with %) [ list<str> | None ]
	    :fields: the list of extra fields to be shown if the result [ list<str> | None ], type ``ami list nomenclature --help`` in the shell to have the list of available fields
	    :order: the list of fields for ordering the result [ list<str> | None ], type ``ami list nomenclature --help`` in the shell to have the list of available fields
	    :limit: limit number of results [ int | tuple<int,int> | None ]
	    :show_archived: show archived results [ bool ]

	kwargs:
	    additionnal constraints field_name:field_value (glob with %), type ``ami list nomenclature --help`` in the shell to have the list of available fields

	Returns:
	    an array of python dictionnaries.
	'''

	if not 'read_status' in kwargs:
		kwargs['read_status'] = 'VALID'

	return pyAMI.utils.smart_execute(client, 'nomenclature', patterns, fields, order, limit, show_archived, **kwargs).get_rows()

#############################################################################

def mode_list_nomenclature(client, args):
	pyAMI_atlas.utils.print_table(client, list_nomenclature(client, **args))

	return 0

#############################################################################
# LIST DATASETS                                                             #
#############################################################################

def list_datasets(client, patterns = None, fields = None, order = None, limit = None, show_archived = False, grl = None, **kwargs):
	'''List ATLAS datasets.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]
	    :patterns: the list of dataset patterns (glob with %) [ list<str> | None ]
	    :fields: the list of extra fields to be shown if the result [ list<str> | None ], type ``ami list datasets --help`` in the shell to have the list of available fields
	    :order: the list of fields for ordering the result [ list<str> | None ], type ``ami list datasets --help`` in the shell to have the list of available fields
	    :limit: limit number of results [ int | tuple<int,int> | None ]
	    :show_archived: show archived results [ bool ]
	    :grl: the Good Run List (GRL) file name [ str | None ]

	kwargs:
	    additionnal constraints field_name:field_value (glob with %), type ``ami list datasets --help`` in the shell to have the list of available fields

	Returns:
	    an array of python dictionnaries.
	'''

	#####################################################################
	# APPLY GRL                                                         #
	#####################################################################

	if grl:

		try:
			import xml.dom.minidom

			doc = xml.dom.minidom.parse(grl)

			kwargs['run_number'] = \
				[node.childNodes[0].data
						for node in doc.getElementsByTagName('Run')]

		except:
			raise pyAMI.exception.Error('could not load GRL `%s`' % grl)

	#####################################################################
	# PATCH PERIODS                                                     #
	#####################################################################

	if 'data_period' in kwargs and kwargs['data_period']:

		tmp = []

		last_char = [
			'0', '1', '2', '3',
			'4', '5', '6', '7',
			'8', '9', '%',
		]

		for data_period in pyAMI.utils.to_array(kwargs['data_period'], sep = ','):

			if data_period:

				if data_period[-1] in last_char:
					tmp.append('%s' % data_period)
				else:
					tmp.append('%s%%' % data_period)

		kwargs['data_period'] = tmp

	#####################################################################
	# PATH PATTERNS                                                     #
	#####################################################################

	patterns = [pattern.rstrip('/') for pattern in pyAMI.utils.to_array(patterns, sep = ',')]

	#####################################################################

	if not 'ami_status' in kwargs:
		kwargs['ami_status'] = 'VALID'

	return pyAMI.utils.smart_execute(client, 'datasets', patterns, fields, order, limit, show_archived, **kwargs).get_rows()

#############################################################################

def mode_list_datasets(client, args):
	pyAMI_atlas.utils.print_table(client, list_datasets(client, **args))

	return 0

#############################################################################
# LIST DATAPERIODS                                                          #
#############################################################################

def list_dataperiods(client, level, year = None):
	'''List ATLAS data periods.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]
	    :level: the period level (1, 2, 3) [ int ]
	    :year: the year [ int | None ]

	Returns:
	    an array of python dictionnaries.
	'''

	command = [
		'AMIListDataPeriods',
		'-periodLevel="%s"' % level,
		'-createdSince="2009-01-01 00:00:00"',
	]

	if year:
		command.append('-projectName="data%02i%%"' % (year % 100))

	return client.execute(command, format = 'dom_object').get_rows()

#############################################################################

def mode_list_dataperiods(client, args):
	pyAMI_atlas.utils.print_table(client, list_dataperiods(client, args['level'], args['year']))

	return 0

#############################################################################
# LIST RUNS                                                                 #
#############################################################################

def list_runs(client, year = None, data_periods = None):
	'''List ATLAS runs.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]
	    :year: the year [ int | None ]
	    :data_periods: the list of data periods [ list<str> | None ]

	Returns:
	    an array of python dictionnaries.
	'''

	command = [
		'AMIListRuns',
	]

	if year:
		command.append('-projectName="data%02i%%"' % (year % 100))

	result = []

	if data_periods:
		command.append('')

		for data_period in data_periods:
			command[-1] = '-period="%s"' % data_period

			result.extend(client.execute(command, format = 'dom_object').get_rows())

	else:
		result.extend(client.execute(command, format = 'dom_object').get_rows())

	return result

#############################################################################

def mode_list_runs(client, args):
	#####################################################################
	#                                                                   #
	#####################################################################

	runs = list_runs(client, args['year'], args['data_periods'])

	#####################################################################
	#                                                                   #
	#####################################################################

	if not args['long']:

		for run in sorted([run['runNumber'] for run in runs]):
			print(run)

		return 0

	#####################################################################
	# PRINT TABLE                                                       #
	#####################################################################

	pyAMI_atlas.utils.print_table(client, runs)

	#####################################################################

	return 0

#############################################################################
# LIST FILES                                                                #
#############################################################################

def list_files(client, dataset, total = None, limit = None):
	'''List dataset files.

	Args:
	    :client: the pyAMI client [ pyAMI.client.Client ]
	    :dataset: the logical dataset name [ str ]
	    :total: produce a grand total [ bool | None]
	    :limit: limit number of results [ int | tuple<int,int> | None ]

	Returns:
	    an array of python dictionnaries.
	'''

	command = [
		'AMIListFiles',
		'-logicalDatasetName="%s"' % dataset,
	]

	if total:
		command.append('-total=yes')
	else:
		command.append('-total=no')

	if limit:

		if isinstance(limit, (list, tuple)):
			command.append('-limit=%i,%i' % (limit[0], limit[1]))
		else:
			command.append('-limit=0,%i' % limit)

	return client.execute(command, format = 'dom_object').get_rows()

#############################################################################

def mode_list_files(client, args):
	#####################################################################
	#                                                                   #
	#####################################################################

	datasets = list_files(client, args['dataset'], total = args['count'], limit = args['limit'])

	#####################################################################
	#                                                                   #
	#####################################################################

	if not args['long']:

		for dataset in datasets:
			print(dataset['LFN'])

		return 0

	#####################################################################
	#                                                                   #
	#####################################################################

	if args['human']:

		for dataset in datasets:
			dataset['events'] = pyAMI_atlas.utils.humanize_numbs(dataset['events'])
			dataset['fileSize'] = pyAMI_atlas.utils.humanize_bytes(dataset['fileSize'])

	#####################################################################
	# PRINT TABLE                                                       #
	#####################################################################

	pyAMI_atlas.utils.print_table(client, datasets)

	#####################################################################

	return 0

#############################################################################

def getAvailableFieldsForCatalog(catalog):
	'''List dataset files.

	Args:
	    :catalog:

	Returns:
	    an array of python dictionnaries.
	'''

	return [field for field in pyAMI.config.tables['datasets'].keys() if field[0] != '@']

#############################################################################
