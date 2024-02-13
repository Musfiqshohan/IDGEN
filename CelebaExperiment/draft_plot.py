



egsde= {'Wearing_Lipstick': 81.60377358490565,
 'Heavy_Makeup': 68.27830188679245,
 'Arched_Eyebrows': 45.75471698113208,
 'Oval_Face': 41.863207547169814,
 'High_Cheekbones': 36.556603773584904,
 'Attractive': 36.20283018867924,
 'Wearing_Earrings': 33.9622641509434,
 'No_Beard': 27.830188679245282,
 'Young': 26.41509433962264,
}





for key in egsde:
	egsde[key] = round(egsde[key],2)

#
#
labels=list(egsde.keys())
# labels
egsde




import numpy as np
import matplotlib.pyplot as plt
import os




		# ax.text(i+d,y[i]+1, y[i], color='red',
        # bbox=dict(facecolor='none', edgecolor='red'))




# addlabels(x, y)

# set width of bar
barWidth = 0.400
fig = plt.subplots(figsize =(12, 8))

# set height of bar
star_list = list(egsde.values())
egsde_list = list(egsde.values())



import numpy as np
import matplotlib.pyplot as plt
import os


# set width of bar
barWidth = 0.400
fig = plt.subplots(figsize =(12, 8))


# Set position of bar on X axis
br1 = np.arange(len(egsde_list))


plt.rcParams['font.size'] = 18


plt.bar(br1, egsde_list,  width = barWidth)

def addlabels(x,y,d):

	for i in range(len(x)):
		plt.text(i+d, y[i]+1, f'{y[i]}%', ha = 'center', rotation=0)

addlabels(br1, egsde_list, d=0.0)


# Adding Xticks
# plt.xlabel('Facial attributes in CelebA dataset', fontweight ='bold', fontsize = 35)
plt.ylabel('Attribute appearance (%)', fontweight ='bold', fontsize = 20)
plt.xticks([r for r in range(len(star_list))],
		labels, rotation=30, fontsize = 18)

plt.title('CelebA attribute change in multi-domain image-to-image translation',fontsize = 22)
plt.tight_layout()

plt.legend()

plt.show()
