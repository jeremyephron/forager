from django import forms

class StyledForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super(StyledForm, self).__init__(*args, **kwargs)
        for visible in self.visible_fields():
            if visible.name == 'photo':
                continue
            visible.field.widget.attrs['class'] = 'form_input_box'


class CreateDatasetForm(StyledForm):
    name = forms.CharField(max_length=300)
    path = forms.CharField(max_length=1000)
