# You Rock My World — para Catalina, mi estrellita ✦

Lyric video (1920x1080, 30 fps) con la canción de Michael Jackson, las fotos editadas
como polaroids, corazones negros y naranjos flotando al ritmo (95 BPM) y una dedicatoria.

## Estructura
- `render.py` — genera el video completo (Python + Pillow + NumPy + ffmpeg).
- `lyrics.json` — las frases y en qué compás aparecen. Se puede editar libremente.
- `assets/photos` — fotos originales (se editan automáticamente: color, contraste, B/N cálido).
- `assets/audio` — la canción.
- `assets/fonts` — Great Vibes, Cormorant Garamond, Montserrat (licencia OFL).

## Generar
```bash
python3 render.py                        # video completo -> out/catalina_you_rock_my_world.mp4
python3 render.py --stills 30,70,260     # imágenes de prueba
python3 render.py --start 60 --dur 10    # clip corto de prueba
```
