### Proof of concept for regular expression based string replacement 

import re

rate_law_str = 'k1*&A1/(Km1 + &A10)/&A1p'
archtype_name = '&A1'
replacement_name = 'B1'

# Updated regex pattern without look-behind, to match `archtype_name` only if it is surrounded by non-word characters or is at the start/end
pattern = r'(?<!\w)' + re.escape(archtype_name) + r'(?!\w)'

print('initial rate law:', rate_law_str)
print('archtype name:', archtype_name)
print('replacement name:', replacement_name)
rate_law_str = re.sub(pattern, replacement_name, rate_law_str)
print('after replacement', rate_law_str)
expected_output = 'k1*B1/(Km1 + &A10)/&A1p'
print(expected_output, rate_law_str, expected_output == rate_law_str)

rate_law_str = 'k1*&A1/(Km1 + &A10)/&A1p'
archtype_name = '&A10'
replacement_name = 'B1'

# Updated regex pattern without look-behind, to match `archtype_name` only if it is surrounded by non-word characters or is at the start/end
pattern = r'(?<!\w)' + re.escape(archtype_name) + r'(?!\w)'

print('initial rate law:', rate_law_str)
print('archtype name:', archtype_name)
print('replacement name:', replacement_name)
rate_law_str = re.sub(pattern, replacement_name, rate_law_str)
print('after replacement', rate_law_str)
expected_output = 'k1*&A1/(Km1 + B1)/&A1p'
print(expected_output, rate_law_str, expected_output == rate_law_str)

rate_law_str = 'k1*&A1/(Km1 + &A10)/&A1p'
archtype_name = '&A1p'
replacement_name = 'B1p'

# Updated regex pattern without look-behind, to match `archtype_name` only if it is surrounded by non-word characters or is at the start/end
pattern = r'(?<!\w)' + re.escape(archtype_name) + r'(?!\w)'

print('initial rate law:', rate_law_str)
print('archtype name:', archtype_name)
print('replacement name:', replacement_name)
rate_law_str = re.sub(pattern, replacement_name, rate_law_str)
print('after replacement', rate_law_str)
expected_output = 'k1*&A1/(Km1 + &A10)/B1p'
print(expected_output, rate_law_str, expected_output == rate_law_str)


