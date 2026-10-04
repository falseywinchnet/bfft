#!/usr/bin/env python3
"""
Patch /tmp/atan2f_ulp_check.cpp so the BFFT f32 path receives a magnitude
consistent with the rounded float x/y inputs.

Before:
    const float mag = static_cast<float>(r);

After:
    const float mag = std::sqrt(x * x + y * y);

Also adds worst-case sample reporting for the BFFT ULP max.
"""

from pathlib import Path

path = Path("/tmp/atan2f_ulp_check.cpp")
if not path.exists():
    raise SystemExit(f"not found: {path}")

s = path.read_text()

old = "const float mag = static_cast<float>(r);"
new = "const float mag = std::sqrt(x * x + y * y);"
if old not in s:
    print("magnitude line already patched or not found")
else:
    s = s.replace(old, new)

# Add worst-case tracking if not already present.
needle = "double max_bfft_abs = 0.0;\n"
insert = """double max_bfft_abs = 0.0;

    float worst_bfft_x = 0.0f;
    float worst_bfft_y = 0.0f;
    float worst_bfft_ref = 0.0f;
    float worst_bfft_val = 0.0f;
"""
if needle in s and "worst_bfft_x" not in s:
    s = s.replace(needle, insert)

old_block = """        max_std_ulp = std::max(max_std_ulp, ulp_f32(stdv, ref));
        max_bfft_ulp = std::max(max_bfft_ulp, ulp_f32(bfftv, ref));

        max_std_abs = std::max(max_std_abs, std::abs((double)stdv - (double)ref));
        max_bfft_abs = std::max(max_bfft_abs, std::abs((double)bfftv - (double)ref));
"""
new_block = """        const uint32_t std_ulp = ulp_f32(stdv, ref);
        const uint32_t bfft_ulp = ulp_f32(bfftv, ref);

        max_std_ulp = std::max(max_std_ulp, std_ulp);
        if (bfft_ulp > max_bfft_ulp) {
            max_bfft_ulp = bfft_ulp;
            worst_bfft_x = x;
            worst_bfft_y = y;
            worst_bfft_ref = ref;
            worst_bfft_val = bfftv;
        }

        max_std_abs = std::max(max_std_abs, std::abs((double)stdv - (double)ref));
        max_bfft_abs = std::max(max_bfft_abs, std::abs((double)bfftv - (double)ref));
"""
if old_block in s and "const uint32_t bfft_ulp" not in s:
    s = s.replace(old_block, new_block)

old_print = """    std::printf("bfft f32    max abs: %.9g rad\\n", max_bfft_abs);
"""
new_print = """    std::printf("bfft f32    max abs: %.9g rad\\n", max_bfft_abs);
    std::printf("bfft worst ulp sample: x=%.9g y=%.9g ref=%.9g val=%.9g diff=%.9g\\n",
                worst_bfft_x,
                worst_bfft_y,
                worst_bfft_ref,
                worst_bfft_val,
                static_cast<double>(worst_bfft_val) - static_cast<double>(worst_bfft_ref));
"""
if old_print in s and "bfft worst ulp sample" not in s:
    s = s.replace(old_print, new_print)

path.write_text(s)
print(f"patched {path}")
