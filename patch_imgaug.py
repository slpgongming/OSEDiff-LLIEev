path = r'C:\Users\Sleep\.conda\envs\osediff\lib\site-packages\imgaug\imgaug.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace(
    'NP_FLOAT_TYPES = set(np.sctypes["float"])',
    'NP_FLOAT_TYPES = {np.float16, np.float32, np.float64}'
).replace(
    'NP_INT_TYPES = set(np.sctypes["int"])',
    'NP_INT_TYPES = {np.int8, np.int16, np.int32, np.int64}'
).replace(
    'NP_UINT_TYPES = set(np.sctypes["uint"])',
    'NP_UINT_TYPES = {np.uint8, np.uint16, np.uint32, np.uint64}'
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('Patched OK — 3 replacements made')
