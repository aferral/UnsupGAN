from pandas import ExcelWriter

import pandas as pd

import os



fbase = 'imagenesTest'

listaFolders = os.listdir(fbase)

listaFolders.remove('test')

listaFolders.remove('old')



for folder in listaFolders: #Tabla

	dataset = folder

	listaFiles = os.listdir(os.path.join(fbase,folder))

	data = {}

	for file in listaFiles:

		if file.split('.')[-1] == 'txt':

			dataTransform = file.split()[2]

			methodName = " ".join(file.split()[3:])

			

			

			with open(os.path.join(fbase,folder,file),'r') as f:	

				texto = f.read() 

				ultimas = texto.split('\n')[-7:]

			print ultimas

			ari = float(ultimas[0].split()[-1])
			dtranf[methodName+'ari'] = ari

			nmi = float(ultimas[1].split()[-1])
			dtranf[methodName+'nmi'] = nmi

			pred = float(ultimas[2].split()[-1])
			dtranf[methodName+'pred'] = pred

			meanCscore  = float(ultimas[3].split()[-1])
			dtranf[methodName+'meanCscore'] = meanCscore

			stdCscore  = float(ultimas[4].split()[-1]) 
			dtranf[methodName+'stdCscore'] = stdCscore

			pcavar  = (ultimas[4].split('variance')[-1])
			dtranf[methodName+'stdCscore'] = stdCscore

			

			print (ari,nmi,pred,meanCscore,stdCscore,pcavar)

			if data.has_key(dataTransform):

				dtranf = data[dataTransform]

				dtranf[methodName] = (ari,nmi,pred,meanCscore,stdCscore,pcavar)

			else:

				data[dataTransform] = {}

				dtranf = data[dataTransform]

				dtranf[methodName] = (ari,nmi,pred,meanCscore,stdCscore,pcavar)

				

			

	df = pd.DataFrame(data)

	writer = ExcelWriter(os.path.join(fbase,dataset+'.xls'))

	df.to_excel(writer,'hoja1')

	writer.save()


