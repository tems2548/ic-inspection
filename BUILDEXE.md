# Build ตัวติดตั้ง IC Inspection (.exe)

คู่มือนี้ใช้ build `IC-Inspection-Setup.exe` สำหรับส่งให้คนอื่นติดตั้งแอปบนเครื่องที่ไม่มี Python หรือ conda

ใช้คำสั่งเดียว:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_app.ps1
```

---

## 1. เครื่องที่ build ต้องมี

| สิ่งที่ต้องมี | หมายเหตุ |
|---|---|
| Windows 10/11 64-bit | |
| Miniconda หรือ Anaconda | ต้องอยู่ที่ `%USERPROFILE%\miniconda3`, `%USERPROFILE%\anaconda3`, `%LOCALAPPDATA%\miniconda3` หรือ `%ProgramData%\miniconda3` ถ้าเคยรัน `setup_env.ps1` แล้วถือว่ามี |
| อินเทอร์เน็ต | รอบแรกดาวน์โหลดราว 1 GB |
| พื้นที่ว่างราว 10 GB | สำหรับ env สำหรับ build, ไฟล์ build และตัวติดตั้ง |
| model ที่ train แล้ว | `results\ocr\crnn_best.pt` และ `results\patchcore\Patchcore\ic\vN\weights\lightning\model.ckpt` |
| weight ของ backbone | `%USERPROFILE%\.cache\huggingface\hub\models--timm--wide_resnet50_2.racm_in1k` มีอยู่แล้วถ้าเคย train หรือรัน PatchCore ในเครื่องนี้ |

ไม่ต้องติดตั้ง PyInstaller หรือ Inno Setup เอง สคริปต์ติดตั้งให้

## 2. ขั้นตอน

1. ดึงโค้ดล่าสุด

   ```powershell
   cd <โฟลเดอร์โปรเจกต์>
   git pull
   ```

2. ปิดแอป IC Inspection ที่เปิดอยู่ และอย่าเปิดโฟลเดอร์ `dist\` ค้างไว้ใน File Explorer หรือ terminal ไม่อย่างนั้นจะ build ไม่ผ่าน (ดูข้อ 6)

3. รัน build

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\build_app.ps1
   ```

   ถ้าต้องการเลือก model ให้ดูตัวเลือกในข้อ 4

4. รอจนขึ้นข้อความ `Done.` สีเขียว
   - รอบแรกใช้เวลาราว 20–30 นาที เพราะต้องสร้าง env `icinspect-build`
   - รอบถัดไปใช้เวลาราว 10 นาที

## 3. ไฟล์ที่ได้

```
dist\
├── IC-Inspection-Setup.exe     ← ส่งไฟล์นี้ให้ผู้ใช้
├── IC Inspection\              ← ตัวแอป (ลองเปิด IC Inspection.exe บนเครื่องนี้ได้)
└── IC-Inspection.zip           ← มีเฉพาะเมื่อใส่ -Zip
```

ถ้าแอปใหญ่เกินราว 3.5 GB (ใส่ model หลายตัว) ตัวติดตั้งจะแยกเป็น `IC-Inspection-Setup.exe` + `IC-Inspection-Setup-1.bin`, `-2.bin`, ... ต้องส่งให้ผู้ใช้ครบทุกไฟล์ และวางไว้ในโฟลเดอร์เดียวกัน

## 4. ตัวเลือก

| ตัวเลือก | ทำอะไร |
|---|---|
| *(ไม่ใส่อะไร)* | ใส่ PatchCore ที่เลข `vN` สูงสุดตัวเดียว |
| `-PatchcoreVersion "v24"` | ใส่เฉพาะ v24 |
| `-PatchcoreVersion "v24,v19"` | ใส่หลาย version **ตัวแรกที่ระบุจะเป็น default** ตอนเปิดแอป ส่วนตัวอื่นเลือกได้จากช่องเลือก model ในแอป |
| `-PatchcoreVersion all` | ใส่ทุก version (ไฟล์จะใหญ่มาก) |
| `-NoModels` | ไม่ใส่ model เลย ผู้ใช้ต้อง train เอง |
| `-Zip` | ทำ zip แบบพกพาเพิ่มด้วย |
| `-NoInstaller` | ไม่ทำตัวติดตั้ง ได้แค่โฟลเดอร์ `dist\IC Inspection\` (ใช้ตอนทดสอบ จะเร็วขึ้นราว 4 นาที) |

ตัวอย่าง:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_app.ps1 -PatchcoreVersion "v24,v19" -Zip
```

ขนาด model แต่ละตัวไม่เท่ากัน (v8 ราว 1.5 GB, v24 ราว 1.1 GB, v19 ราว 0.4 GB) ใส่เฉพาะตัวที่จะใช้จริง

**สิ่งที่ถูกใส่ในตัวติดตั้งด้วย:** ไฟล์ใน `configs\` ของเครื่องที่ build เช่นค่า `part_score` ใน `configs\patchcore.yaml` ต้องให้ตรงกับ model ที่ใส่ไปก่อน build

## 5. ทดสอบก่อนส่ง

1. เปิด `dist\IC Inspection\IC Inspection.exe` แล้วตรวจรูปสัก 1–2 รูป
2. หรือลองติดตั้งจริงด้วย `dist\IC-Inspection-Setup.exe` แล้วถอนการติดตั้งจาก Settings → Apps

## 6. ปัญหาที่พบบ่อย

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| `PermissionError: Access is denied` หรือ `being used by another process` | มีโปรแกรมเปิด `dist\IC Inspection` ค้างไว้ ให้ปิดแอป IC Inspection, ปิด File Explorer และ terminal ที่อยู่ในโฟลเดอร์นั้น แล้วรันใหม่ |
| `conda not found` | ไม่พบ conda ที่ path มาตรฐาน ให้ติดตั้ง Miniconda หรือรัน `setup_env.ps1` ก่อน |
| `PatchCore vN not found` | ไม่มี version นั้นในเครื่อง ข้อความจะบอกว่ามี version ไหนให้เลือก |
| ขึ้นเตือนสีเหลือง `no timm backbone ...` | เครื่องนี้ไม่มี weight ของ backbone แอปที่ได้จะต้องใช้อินเทอร์เน็ตตอนโหลด PatchCore แก้โดยรัน PatchCore ใน env `icinspect` สักครั้ง (เช่นตรวจรูป 1 รูป หรือ train) แล้ว build ใหม่ |
| `Inno Setup install failed` | ติดตั้ง Inno Setup 6 เองจาก https://jrsoftware.org/isdl.php แล้วรันใหม่ |
| ต้องการเริ่ม env สำหรับ build ใหม่หมด | `conda env remove -n icinspect-build` แล้วรัน `build_app.ps1` ใหม่ |

## 7. ฝั่งผู้ใช้ที่ได้รับไฟล์

1. ดับเบิลคลิก `IC-Inspection-Setup.exe`
2. ถ้าขึ้น "Windows protected your PC" ให้กด **More info → Run anyway** (ไฟล์ไม่ได้เซ็นชื่อดิจิทัล)
3. ใช้โฟลเดอร์ติดตั้งที่ตั้งไว้ให้ (`AppData\Local\Programs\IC Inspection`) ไม่ต้องใช้สิทธิ์ admin
4. เปิดจากไอคอนบน Desktop หรือ Start menu

เครื่องผู้ใช้ต้องเป็น Windows 10/11 64-bit, RAM 8 GB ขึ้นไป, พื้นที่ว่างราว 4 GB ไม่ต้องมีการ์ดจอ และไม่ต้องมีอินเทอร์เน็ต

---

## ภาคผนวก: ไฟล์ที่เกี่ยวข้อง

| ไฟล์ | หน้าที่ |
|---|---|
| `build_app.ps1` | สคริปต์ build หลัก: สร้าง env → PyInstaller → ใส่ configs/model/backbone → Inno Setup |
| `packaging\app_entry.py` | จุดเริ่มของ exe: เปิด GUI และรองรับงานเบื้องหลังของแท็บ Train/Setup (`--run-module`, `--run-script`) และตั้งให้ใช้ backbone ที่ใส่มาแบบ offline |
| `packaging\ic_inspection.spec` | ค่าตั้งของ PyInstaller (package ที่ต้องรวมเข้าไป) |
| `packaging\installer.iss` | สคริปต์ Inno Setup ของตัวติดตั้ง |
| `packaging\constraints.txt` | เลข version ของ library ที่ใช้ตอน build |
| `src\icinspect\config.py`, `src\icinspect\gui\runner.py` | ทำให้โค้ดทำงานได้เมื่อเป็น exe (ตำแหน่งไฟล์ และการเรียกงานเบื้องหลัง) |

แอปใช้ PyTorch แบบ CPU และ build ใน env แยกชื่อ `icinspect-build` env `icinspect` ที่ใช้พัฒนา (CUDA) จึงไม่ถูกแตะ

ถ้าเปลี่ยน version ของ torch, torchvision หรือ anomalib ใน env `icinspect` ให้แก้เลข version ใน `build_app.ps1` และ `packaging\constraints.txt` ให้ตรงกัน แล้วลบ env `icinspect-build` เพื่อให้สร้างใหม่
