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

import sys, textwrap, pyAMI.utils, pyAMI.config

#############################################################################

def _is_not_reserved(field):

	field = field.lstrip('.')

	return field != 'PROCESS'       \
	       and                      \
	       field != 'PROJECT'       \
	       and                      \
	       field != 'AMIELEMENTID'  \
	       and                      \
	       field != 'AMIENTITYNAME'

#############################################################################

def print_table(client, table, hsep = ' | ', vsep = '-', wrap_width = 60, stream = sys.stdout):

	if client.config.format == 'json':
		pyAMI.utils.print_json(table)
		return

	#####################################################################
	# REORGANIZE DATA                                                   #
	#####################################################################

	rows = []

	for i, project in enumerate(table):

		if i == 0:
			row = [field for field, value in list(project.items()) if _is_not_reserved(field)]

			rows.append(row)

		row = [value for field, value in list(project.items()) if _is_not_reserved(field)]

		rows.append(row)

	#####################################################################

	if not rows:
		return

	nr_row = len(rows)
	nr_col = len(rows[0])

	if nr_col > 1:
		#############################################################
		# COMPUTE CELL PROPERTIES                                   #
		#############################################################

		heights = nr_row * [0]
		widths = nr_col * [0]

		for i in range(nr_row):

			for j in range(nr_col):
				#############################################
				# WRAP LINE                                 #
				#############################################

				rows[i][j] = textwrap.wrap(rows[i][j], width = wrap_width)

				#############################################
				# COMPUTE HEIGHT                            #
				#############################################

				length = len(rows[i][j])

				if heights[i] < length:
					heights[i] = length

				#############################################
				# COMPUTE WIDTH                             #
				#############################################

				for value in rows[i][j]:

					length = len(value)

					if widths[j] < length:
						widths[j] = length

			for j in range(nr_col):
				#############################################
				# PAD LINE HEIGHT                           #
				#############################################

				rows[i][j] += (heights[i] - len(rows[i][j])) * ['']

		#############################################################
		# COMPUTE TABLE WIDTH                                       #
		#############################################################

		hsep_l = hsep.lstrip()
		hsep_r = hsep.rstrip()

		total_width = sum(widths) + len(hsep_l) + (nr_col - 1) * len(hsep) + len(hsep_r)

		#############################################################
		# PRINT TABLE                                               #
		#############################################################

		for i, row in enumerate(rows):

			if vsep and i == 0:
				stream.write(total_width * vsep + '\n')

			for j in range(heights[i]):
				stream.write(hsep_l + hsep.join(['%-*s' % (widths[k], fields[j]) for k, fields in enumerate(row)]) + hsep_r + '\n')

			if vsep and 0 == 0:
				stream.write(total_width * vsep + '\n')

		#############################################################

	else:
		#############################################################
		# PRINT TABLE                                               #
		#############################################################

		for row in rows[1: ]:
			for field in row[0: ]:
				stream.write(field + '\n')

#############################################################################

def print_list(client, table, title = None, stream = sys.stdout):

	if client.config.format == 'json':
		pyAMI.utils.print_json(table)
		return

	for fields in table:
		#############################################################
		# COMPUTE MAX                                               #
		#############################################################

		MAX = 0

		for field in fields:

			if field != title:
				length = len(field)

				if MAX < length:
					MAX = length

		#############################################################
		# PRINT LIST                                                #
		#############################################################

		if title:
			stream.write('%s: %s\n' % (title, fields[title]))

			SHIFT = '  '
		else:
			SHIFT = ''

		for field, value in list(fields.items()):

			if field != title:
				stream.write('%s%-*s: %s\n' % (SHIFT, MAX, field, value.strip().replace('\n', '\\n')))

#############################################################################

def humanize_bytes(num):

	num = float(num)

	for x in ['B.', 'KB', 'MB', 'GB', 'TB', 'PB']:

		if num < 1024.0:
			return '%3.2f %s' % (num, x)

		num /= 1024.0

#############################################################################

def humanize_numbs(num):

	result = []

	for i, c in enumerate(reversed(num)):

		if i and (i % 3) == 0:
			result.insert(0, ',')

		result.insert(0, c)

	return ''.join(result)

#############################################################################
