import os
from pdf2image import convert_from_path

for x in os.walk('.'):
    for y in x[2]:
        if y.endswith('.pdf'):
            p = os.path.join(x[0], y)
            convert_from_path(p, dpi=200, fmt='png', paths_only=True, single_file=True)[0].save(p+".png")