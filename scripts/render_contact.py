from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
root=Path(__file__).resolve().parents[1]/'tmp/qa'
pages=sorted(root.glob('page-*.png'))
for i in range(0,len(pages),8):
    sheet=Image.new('RGB',(1800,1320),'white')
    for j,p in enumerate(pages[i:i+8]):
        im=Image.open(p);x=(j%4)*450;y=(j//4)*660
        sheet.paste(ImageOps.contain(im,(440,630)),(x,y+25))
        ImageDraw.Draw(sheet).text((x+8,y+4),f'Page {i+j+1}',fill='black')
    sheet.save(root/f'contact-{i//8}.png')

