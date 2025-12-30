import * as React from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { DialogFooter } from '@/components/ui/dialog';
import { Loader2 } from 'lucide-react';

export type ConfigField = {
  id: string;
  label: string;
  value: string;
  onChange: (nextValue: string) => void;
  placeholder?: string;
  helperText?: string;
  inputProps?: Omit<React.ComponentProps<typeof Input>, 'value' | 'onChange' | 'id'>;
  transform?: (value: string) => string;
};

interface ConfigFormProps {
  fields: ConfigField[];
  onSubmit: () => void;
  submitLabel: string;
  isSubmitting?: boolean;
  isValid?: boolean;
}

export function ConfigForm({
  fields,
  onSubmit,
  submitLabel,
  isSubmitting = false,
  isValid = true,
}: ConfigFormProps) {
  return (
    <div className="space-y-4">
      {fields.map((field) => {
        const { inputProps, transform } = field;
        const { className, ...restProps } = inputProps || {};
        return (
          <div key={field.id} className="space-y-2">
            <Label htmlFor={field.id}>{field.label}</Label>
            <Input
              id={field.id}
              value={field.value}
              placeholder={field.placeholder}
              className={className}
              onChange={(event) => {
                const nextValue = transform
                  ? transform(event.target.value)
                  : event.target.value;
                field.onChange(nextValue);
              }}
              {...restProps}
            />
            {field.helperText && (
              <p className="text-xs text-muted-foreground">{field.helperText}</p>
            )}
          </div>
        );
      })}

      <DialogFooter>
        <Button
          type="button"
          onClick={onSubmit}
          disabled={!isValid || isSubmitting}
        >
          {isSubmitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          {submitLabel}
        </Button>
      </DialogFooter>
    </div>
  );
}
