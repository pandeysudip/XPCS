from xpcs_class import XPCS

g2=XPCS(temp, light, dark,roi_edge=(710,730), num_levels=8, num_bufs=12)

#single temp plot
temp='80K'
light=
dark=[]
light_offr=
dark_offr=[]
light2=
dark2=[]
g2.total_intensity()
g2.intensity_q(light_offr, dark_offr)
g2.sg_g2()
g2.sg_g2_both(light_offr, dark_offr)
g2.intensity_corr(light2, dark2,frame=(500, 4000, 500))
g2.intensity_corr_small(light2, dark2, xrange=(720,760),yrange=(425,525),frame=(500, 4000, 500))

#to plot all
temp_all=['80K', '65K', '55K', '50K', '40K', '300K']
light=[]
dark=[[], []]
light_offr=[]
dark_offr=[[], []]
light2=[]
dark2=[[], []]
for i in range(len(temp_all)):
    g2=XPCS(temp_all[i], light[i], dark,roi_edge=(710,730), num_levels=8, num_bufs=12)
    g2.total_intensity()
    g2.intensity_q(light_offr[i], dark_offr[i])
    g2.sg_g2()
    g2.sg_g2_both(light_offr[i], dark_offr[i])
    g2.intensity_corr(light2[i], dark2[i],frame=(500, 4000, 500))
    g2.intensity_corr_small(light2[i], dark2[i], xrange=(720,760),yrange=(425,525),frame=(500, 4000, 500))