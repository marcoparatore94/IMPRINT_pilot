# utils/info_panel.py
def get_info_html():
    return """
    <h4>IMPRINT – Informazioni</h4>
    <ul>
      <li><b>Feature ecografiche:</b> diametro >8 cm, margini irregolari, color score 4, assenza di ombre.</li>
      <li><b>Markers:</b> SII, SIRI, NLR, PIV. Cut-off pilota (Youden): SII ~700, SIRI ~0.96, NLR ~2.5, PIV ~275.</li>
      <li><b>Classi di rischio:</b> basso &lt;0.4%, intermedio 0.4–2.2%, alto ≥2.3% (ispirate al MYLUNAR).</li>
      <li><b>Uso clinico:</b> algoritmo di triage; non sostituisce il giudizio clinico.</li>
      <li><b>Limitazioni:</b> modello addestrato con controlli benigni simulati; ricalibrare con dati reali.</li>
    </ul>
    """