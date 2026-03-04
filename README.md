# video-course-analyzer

video-course-analyzer, video ders içeriklerini işleyip analiz eden; API üzerinden iş tetikleyen, CPU/GPU worker’larıyla transkripsiyon/özetleme adımlarını çalıştıran ve RAG servisiyle içerik sorgulaması sağlayan çok servisli bir başlangıç iskeletidir.

## Servisler
- `api`: İş kabulü, durum sorgulama ve orkestrasyon uçları.
- `worker-cpu`: CPU tabanlı görevlerin tüketimi ve işlenmesi.
- `worker-gpu`: GPU hızlandırmalı model görevleri (opsiyonel profil).
- `rag`: Vektörleme/geri getirme ve sorgu katmanı.

## Fast Mode MVP Akışı
1. `api` yeni bir analiz işi alır ve `data/jobs` altında iş kaydı oluşturur.
2. İş, öncelikle `worker-cpu` tarafından temel ön işleme/transkripsiyon adımlarından geçirilir.
3. GPU gerektiren adımlar varsa `worker-gpu` devreye girer (profil etkinse).
4. Üretilen çıktılar `rag` tarafından indekslenir ve sorgulanabilir hale getirilir.
5. `api` iş durumunu ve nihai özet/çıktıları istemciye sunar.
