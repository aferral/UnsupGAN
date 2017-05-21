from pandas import ExcelWriter
import pandas as pd
import os



fbase = 'imagenesTest'
listaFolders = os.listdir(fbase)
listaFolders.remove('test')
listaFolders.remove('old')


for folder in listaFolders: #Tabla
	if os.path.isdir(os.path.join(fbase,folder)):
		dataset = folder
		listaFiles = os.listdir(os.path.join(fbase,folder))
		writer = ExcelWriter(os.path.join(fbase, dataset + '.xls'))
		hoja = 0
		data = {}
		for file in listaFiles:
			if file.split('.')[-1] == 'txt':

				dataTransform = file.split()[2]
				methodName = " ".join(file.split()[3:]).replace('.txt','')

				temp = {}

				with open(os.path.join(fbase,folder,file),'r') as f:
					texto = f.read()
					ultimas = texto.split('\n')[-7:]
				print ultimas

				ari = float(ultimas[0].split()[-1])
				temp['ari'] = ari

				nmi = float(ultimas[1].split()[-1])
				temp['nmi'] = nmi

				pred = float(ultimas[2].split()[-1])
				temp['pred'] = pred

				meanCscore  = float(ultimas[3].split()[-1])
				temp['meanCscore'] = meanCscore

				stdCscore  = float(ultimas[4].split()[-1])
				temp['stdCscore'] = stdCscore

				pcavar  = sum(map(float,(ultimas[5].split('variance[')[-1])[:-1].split()))
				temp['pcavar'] = pcavar

				data[str(dataTransform) + "_" + str(methodName)] = temp

		df = pd.DataFrame(data).transpose()
		df.to_excel(writer,'hoja0')
		hoja+=1
		writer.save()


