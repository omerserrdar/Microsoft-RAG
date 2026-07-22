"""
core — Project Antigravity: Ana Modül Paketi
=============================================
Bu paket, yerel RAG (Retrieval-Augmented Generation) sisteminin
tüm çekirdek bileşenlerini barındırır.

Bileşenler:
    - database  : Asenkron SQLite veritabanı ve Cosine Similarity UDF
    - ingester  : Döküman okuma, akıllı parçalama ve vektörleştirme
    - retriever : Vektör benzerlik araması ve getirme (retrieval)
    - generator : LLM streaming yanıt üretimi (generation)
"""

