from __future__ import print_function, division

'''
Trigger definition class

'by_key' and 'values' member variables are automatically created when instantiated.


Kyuhwa Lee, 2015
Swiss Federal Institute of Technology Lausanne (EPFL)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''

class TriggerDefTemplate:
	def __init__(self):
		for attr in dir(self):
			if hasattr(self, 'by_value')==False:
				self.by_key= {}
				self.by_value= {}
			if not callable(getattr(self,attr)) and not attr.startswith("__"):
				#print(attr, getattr(self,attr))
				self.by_key[attr]= getattr(self,attr)
				self.by_value[getattr(self,attr)]= attr

