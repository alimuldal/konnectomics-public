# Copyright 2014 Alistair Muldal <alistair.muldal@pharm.ox.ac.uk>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from sys import stdout
from time import time

class Waitbar(object):
	def __init__(	self, amount=0., barwidth=50, totalwidth=75, title=None,
			showETA=True):

		self.showETA = showETA
		self._bw = barwidth
		self._tw = totalwidth
		self._stime = time()
		self._bar = ''
		self._eta = ''

		if title is not None:
			print title

		self.update(amount)

	def update(self,newamount):

		self._done = newamount

		n = int(round(self._done * self._bw))
		# bar = u'\u25AE'*n
		bar = '#' * n
		pad = '-' * (self._bw - n)
		self._bar = '[' + bar + pad + '] %2i%%' %(self._done * 100.)

		if self.showETA:
			if self._done == 0:
				self._eta = '  ETA: ?'
			else:
				dt = time() - self._stime
				eta = (dt / self._done) * (1. - self._done)
				self._eta = '  ETA: %s' %s2h(eta)
		self.display()

	def display(self):
		stdout.write('\r' + ' '*self._tw)
		if self._done >= 1.:
			ftime = s2h(time() - self._stime)
			stdout.write('\r--> Completed: %s\n' %ftime)
		else:
			nspace = max(
				(0,self._tw - (len(self._bar) + len(self._eta)))
				)
			stdout.write('\r' + self._bar + self._eta + ' '*nspace)

		stdout.flush()

class ElapsedTimer(object):

	def __init__(self, title=None, width=75):
		self.title = title
		self._stime = time()
		self._width = width
		npad = width - len(title)
		stdout.write('\r' + self.title + ' '*npad)
		stdout.flush()

	def done(self):
		elapsed = s2h(time() - self._stime)
		donestr = 'done: ' + elapsed + '\n'
		npad = self._width - (len(self.title) + len(donestr))
		stdout.write('\r' + self.title + ' '*npad + donestr)
		stdout.flush()

def s2h(ss):
	mm,ss = divmod(ss,60)
	hh,mm = divmod(mm,60)
	dd,hh = divmod(hh,24)
	tstr = "%02i:%04.1f" %(mm,ss)
	if hh > 0:
		tstr = ("%02i:" %hh) + tstr
	if dd > 0:
		tstr = ("%id " %dd) + tstr
	return tstr