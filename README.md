# PENERAPAN-METODE-BACKGROUND-SUBTRACTION-MENGGUNAKAN-GAUSSIAN-MIXTURE-MODEL-(GMM)-DAN-LOGISTIC-REGRESSION-DALAM-MENDETEKSI-PERGERAKAN-DAN-MENGGOLONGKAN-KENDARAAN-PADA-JALAN-TOL
&emsp;&emsp;Program ini adalah implementasi skripsi yang berjudul "PENERAPAN METODE *BACKGROUND SUBTRACTION* MENGGUNAKAN *GAUSSIAN MIXTURE MODEL* (GMM) DAN *LOGISTIC REGRESSION* DALAM MENDETEKSI PERGERAKAN DAN MENGGOLONGKAN KENDARAAN PADA JALAN TOL"  dimana ini dibuat dengan tujuan dapat membantu para ahli dalam menghitung jumlah kendaraan yang lewat pada jalan bebas hambatan(tol) sebagai data statistik untuk mengukur kenaikan volume lalu lintas, memperkirakan pendapatan dari pemakaian jalan, atau pengembangan jalan, serta juga mengklasifikasi jenis kendaraan yang dapat digunakan untuk mengambil keputusan jumlah gerbang keluar pada jenis mobil yang diperlukan dan mencegah terjadinya kemacetan pada jalan tol dan bahan pertimbangan untuk memperluas jalan tol.<br/><br/>
Ucapan terima kasih kepada Bapak Mula’ab, S.Si., M.Kom. dan Ibu Dr. Indah Agustien S, S.Kom, M.Kom. selaku Dosen Pembimbing saya.<br/>
Nama&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;: Anggi Shofiyuddin<br/>
NIM&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;: 160411100114<br/>
Mata kuliah&emsp;&emsp;&emsp;&emsp;&ensp;: Skripsi<br/>
Jurusan&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;: Teknik Informatika<br/>
Perguruan Tinggi&emsp;&emsp;: Universitas Trunojoyo Madura<br/><br/>
***Environment***<br/>
Program ini dijalankan menggunakan bahasa Python, dengan *library*:<br/>
* Pandas<br/>
* Numpy<br/>
* cv2<br/>
* PySide<br/>
* Csv<br/>
* Sklearn<br/>
* Sys<br/>
* Os<br/>
* Winsound<br/>
* Shutil<br/>
* Time<br/>
* Math<br/><br/>

**<h2 align="center">Implementasi Aplikasi</h2>**
<table align="center">
  <tr align="center">
    <td>Tampilan Training Logistic Regression</td>
  </tr>
  <tr align="center">
    <td><img src="/Images/TampilanTrainingLogisticRegression.png" width="350"/></td>
  </tr>
</table>

&emsp;&emsp;Tampilan ini adalah tampilan awal *Training Logistic Regression* yang nantinya menginisialisasi *learning rate*, iterasi maksimal, dan nilai *threshold*, serta memasukkan dataset citra kendaraan dari situs *kaggle* dan citra hasil deteksi kendaraan *training* (semua citra kendaraan yang dideteksi, kecuali citra pertama dari setiap objek yang dideteksi) yang telah digolongkan menjadi 4 golongan secara manual.<br/><br/>
<table align="center">
  <tr align="center">
    <td>Tampilan Hasil Training Data Ekstraksi Fitur Citra</td>
  </tr>
  <tr align="center">
    <td><img src="/Images/TampilanHasilTrainingDataEkstraksiFiturCitra.png" width="350"/></td>
  </tr>
</table>

&emsp;&emsp;Tampilan ini adalah tampilan setelah dataset telah melakukan *training* menggunakan metode *Logistic Regression*.<br/><br/>
<table align="center">
  <tr align="center">
    <td>Tampilan Deteksi Kendaraan</td>
  </tr>
  <tr align="center">
    <td><img src="/Images/TampilanDeteksiKendaraan.png" width="350"/></td>
  </tr>
</table>

&emsp;&emsp;Tampilan ini adalah tampilan awal deteksi kendaraan yang nantinya memasukan data video, citra *background*, dan citra *background* biner.<br/><br/>
<table align="center">
  <tr align="center">
    <td>Tampilan Proses dan Hasil Deteksi Kendaraan</td>
  </tr>
  <tr align="center">
    <td><img src="/Images/TampilanProsesdanHasilDeteksiKendaraan1.png" height="273"/><img src="/Images/TampilanProsesdanHasilDeteksiKendaraan2.png" width="350"/></td>
  </tr>
</table>

&emsp;&emsp;Tampilan ini adalah tampilan proses dan hasil deteksi kendaraan yang nantinya akan menghasilkan sebuah *output* berupa video hasil dari proses deteksi kendaraan menggunakan metode *Background Subtraction* dengan *Gaussian Mixture Model*.<br/><br/>

**<h2 align="center">Kesimpulan</h2>**
1.	Dalam melakukan deteksi pergerakan objek kendaraan menggunakan metode *Background Subtraction* dengan *Gaussian Mixture Model*, serta menggunakan 21 data test dapat diketahui memperoleh akurasi yang cukup tinggi yaitu hasil *Precision* sebesar 78.52%, *Recall* sebesar 92.39%, *F1-score* sebesar 84.0% dan *Classification Accuracy* sebesar 73.37%, serta dalam proses deteksi pergerakan kendaraan tersebut tidak banyak melakukan kesalahan deteksi seperti objek kendaraan yang tidak terdeteksi dan pada 1 objek kendaraan yang terdeteksi menjadi lebih dari 1 objek kendaraan, serta lebih dari 1 objek kendaraan yang terdeteksi menjadi 1 objek kendaraan.<br/>
2.	Dan dalam mengklasifikasi atau menggolongkan citra hasil deteksi kendaraan menggunakan metode *Logistic Regression* memperoleh akurasi yang cukup rendah yaitu memperoleh hasil *Precision* sebesar 45.87%, *Recall* sebesar 39.87%, *F1-score* sebesar 34.59% dan *Classification Accuracy* sebesar 46.33% yang disebabkan banyaknya fitur yang tidak relevan dapat mempengaruhi *testing* pada klasifikasi jenis kendaraan dan menyebabkan akurasi yang diperoleh menjadi rendah.
