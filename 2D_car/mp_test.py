import multiprocessing as mp
get_descriped(self)
def joba():
	while True:
		res = 0
		for i in range(100000):
			res+=i+i**2+i**3
		print('job a done')

def jobb():
	while True:
		res = 0
		for i in range(100000):
			res+=i+i**2+i**3+i**3.1564
		print('job b done')

def multicore():
	print('multicore start!')
	p1 = mp.Process(target=joba)
	p2 = mp.Process(target=jobb)
	p1.start()
	p2.start()

multicore()